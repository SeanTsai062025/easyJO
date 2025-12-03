import cv2
import mediapipe as mp
import threading
import time
import numpy as np
from collections import deque

mp_pose = mp.solutions.pose
WINDOW_SIZE = 3     # SMA 的 W

# 全域共享
latest_frame = None        # 最新原始畫面（彩色）
processed_frame = None     # 已畫好 Pose 的畫面（灰階或灰階+骨架）
latest_frame_id = -1       # 目前 latest_frame 是第幾幀
lock = threading.Lock()
running = True

VIDEO_PATH = "test_video1.MP4"
TARGET_FPS = 10.0          # 播放 & 取樣幀率
SAMPLE_FPS = TARGET_FPS    # 真正給頻率分析用的採樣率（由主 thread 控 FPS）

# -------- 只用 1~12 + 23,24 --------
VISIBLE_POINTS = list(range(1, 13)) + [23, 24]

# -------- 頻率分析參數（針對 2~3Hz 的穩定運動） --------
GLOBAL_WINDOW_SECONDS = 4.0                       # 全身頻率用 4 秒 window，比較穩定
GLOBAL_WINDOW_SIZE = int(TARGET_FPS * GLOBAL_WINDOW_SECONDS)

ALPHA = 0.7                                       # displacement 平滑少一點，避免被抹平
MIN_FREQ = 1.0                                    # 你關心 2~3Hz → 限制在 1~4
MAX_FREQ = 4.0

PEAK_RATIO_THRESH = 2.0                           # peak / 平均能量 門檻（越大越嚴格）


def draw_selected_landmarks(image, landmarks):
    """只畫指定的 landmarks 點，不畫線"""
    h, w, _ = image.shape
    for idx in VISIBLE_POINTS:
        lm = landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)


def compute_freq(buffer, fps):
    """
    對一個「全身位移」訊號 buffer 做 FFT，
    回傳 (peak_freq, is_stable)
    - peak_freq: 主要頻率 (Hz) 或 None
    - is_stable: 頻譜上 peak 是否夠明顯
    """
    if len(buffer) < 8:
        return None, False

    sig = np.array(buffer, dtype=np.float32)

    # 去 DC
    sig = sig - sig.mean()
    if np.allclose(sig, 0.0):
        return None, False

    # Hamming window 減少頻譜外洩，讓 peak 穩定一點
    window = np.hamming(len(sig))
    sig_win = sig * window

    fft_vals = np.fft.rfft(sig_win)
    freqs    = np.fft.rfftfreq(len(sig_win), d=1.0 / fps)

    # 只看你 care 的頻帶
    band_idx = np.where((freqs >= MIN_FREQ) & (freqs <= MAX_FREQ))[0]
    if len(band_idx) == 0:
        return None, False

    mag      = np.abs(fft_vals)
    band_mag = mag[band_idx]

    # 找頻帶中最大 peak
    rel_peak_idx = np.argmax(band_mag)
    peak_idx     = band_idx[rel_peak_idx]
    peak_freq    = float(freqs[peak_idx])
    peak_amp     = float(band_mag[rel_peak_idx])

    mean_amp  = float(band_mag.mean() + 1e-6)
    peak_ratio = peak_amp / mean_amp

    is_stable = peak_ratio > PEAK_RATIO_THRESH
    return peak_freq, is_stable


def capture_loop():
    global latest_frame, running, latest_frame_id, SAMPLE_FPS

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Cannot open video")
        running = False
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = TARGET_FPS

    # 只用 frames_to_skip 拉齊時間軸，實際處理幀率由 TARGET_FPS 控制
    if src_fps > TARGET_FPS:
        frames_to_skip = int(src_fps / TARGET_FPS) - 1
    else:
        frames_to_skip = 0

    frame_interval = 1.0 / TARGET_FPS
    SAMPLE_FPS = TARGET_FPS
    print(f"src_fps = {src_fps:.2f}, SAMPLE_FPS = {SAMPLE_FPS:.2f}")

    frame_id  = 0
    last_time = time.time()

    while running:
        ret, frame = cap.read()
        if not ret:
            running = False
            break

        with lock:
            latest_frame    = frame.copy()
            latest_frame_id = frame_id

        frame_id += 1

        # 跳過一些 frame 讓畫面時間同步
        for _ in range(frames_to_skip):
            ret, _ = cap.read()
            if not ret:
                running = False
                break

        # 控制撥放節奏
        now     = time.time()
        elapsed = now - last_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        last_time = time.time()

    cap.release()



def processing_loop():
    global latest_frame, processed_frame, running, latest_frame_id, SAMPLE_FPS

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    SMALL_W = 480
    SMALL_H = 270

    # ====== 狀態變數 ======
    prev_center = None                      # P_curr(t-1)
    window_P = deque(maxlen=WINDOW_SIZE)    # Step3: SMA 視窗 (存 P_curr)
    window_M = deque(maxlen=WINDOW_SIZE)    # Step4: Envelope 視窗 (存 |V|)
    envelope = 0.0                          # E(t)

    last_sample_time = None
    ema_dt = None
    last_processed_id = -1

    # 鏡頭切換閾值：畫面 10%
    cut_thresh = 0.1 * min(SMALL_W, SMALL_H)

    target_radius   = 30.0   # 圓盤半徑 & 動態增益目標振幅
    compass_margin  = 20     # 右下小圈離邊界距離
    SAMPLE_FPS      = TARGET_FPS

    while running:
        frame_copy = None
        cur_id = None

        # 只處理最新一幀
        with lock:
            if latest_frame is not None and latest_frame_id != last_processed_id:
                frame_copy = latest_frame.copy()
                cur_id = latest_frame_id

        if frame_copy is None:
            time.sleep(0.001)
            continue

        last_processed_id = cur_id

        # 縮小 & 轉灰階
        small = cv2.resize(frame_copy, (SMALL_W, SMALL_H))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray_rgb = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2RGB)

        result = pose.process(gray_rgb)

        h, w, _ = gray_bgr.shape

        # 中間大圈圓心（最終 output + 箭頭）
        center_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        # 右下小圈圓心（原始未投影 output）
        compass_center = np.array(
            [w - compass_margin - target_radius,
             h - compass_margin - target_radius],
            dtype=np.float32
        )

        fusion_center = None   # P_curr(t)
        face_point    = None   # 臉（鼻子）
        pelvis_center = None   # 骨盆中點

        # ====== Step 1: 訊號融合 + 頭 / 骨盆 ======
        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            draw_selected_landmarks(gray_bgr, result.pose_landmarks)

            # 14 點平均 = P_curr
            xs, ys = [], []
            for idx in VISIBLE_POINTS:
                lm = lms[idx]
                x_px = lm.x * SMALL_W
                y_px = lm.y * SMALL_H
                if 0 <= x_px < SMALL_W and 0 <= y_px < SMALL_H:
                    xs.append(x_px)
                    ys.append(y_px)
            if xs and ys:
                cx = float(np.mean(xs))
                cy = float(np.mean(ys))
                fusion_center = np.array([cx, cy], dtype=np.float32)

            # 臉：鼻子 (0)
            nose_lm = lms[0]
            nx = nose_lm.x * SMALL_W
            ny = nose_lm.y * SMALL_H
            if 0 <= nx < SMALL_W and 0 <= ny < SMALL_H:
                face_point = np.array([nx, ny], dtype=np.float32)

            # 骨盆：左右臀 (23,24) 平均
            lh_lm = lms[11]
            rh_lm = lms[12]
            lx, ly = lh_lm.x * SMALL_W, lh_lm.y * SMALL_H
            rx, ry = rh_lm.x * SMALL_W, rh_lm.y * SMALL_H
            if (0 <= lx < SMALL_W and 0 <= ly < SMALL_H and
                0 <= rx < SMALL_W and 0 <= ry < SMALL_H):
                pelvis_center = np.array([(lx + rx) / 2.0,
                                          (ly + ry) / 2.0],
                                         dtype=np.float32)

        # 動態輸出向量（2D）
        scaled_V = np.array([0.0, 0.0], dtype=np.float32)

        # arrow 用的單位方向向量（骨盆→頭）
        dir_u = None

        # ====== Step 2~5：有人體才做 ======
        if fusion_center is not None:
            now = time.time()

            # FPS 估計
            if last_sample_time is not None:
                dt = now - last_sample_time
                if dt > 1e-4:
                    if ema_dt is None:
                        ema_dt = dt
                    else:
                        ema_dt = 0.9 * ema_dt + 0.1 * dt
            last_sample_time = now

            if ema_dt is not None and ema_dt > 1e-4:
                SAMPLE_FPS = 1.0 / ema_dt
            else:
                SAMPLE_FPS = TARGET_FPS

            # Step 2：Cut detection
            if prev_center is not None:
                dist = float(np.linalg.norm(fusion_center - prev_center))
                if dist > cut_thresh:
                    window_P.clear()
                    window_M.clear()
                    envelope = 0.0
            prev_center = fusion_center.copy()

            # Step 3：SMA baseline
            window_P.append(fusion_center.copy())
            if len(window_P) > 0:
                baseline = np.mean(window_P, axis=0)
            else:
                baseline = fusion_center.copy()

            V = fusion_center - baseline

            # Step 4：Envelope（sliding window max）
            M_inst = float(np.linalg.norm(V))
            window_M.append(M_inst)
            envelope = max(window_M) if len(window_M) > 0 else 0.0

            # Step 5：Dynamic gain
            if envelope < 1e-4:
                scaled_V = np.array([0.0, 0.0], dtype=np.float32)
            else:
                gain = target_radius / envelope
                scaled_V = V * gain

            # 骨盆→頭方向 dir_u
            if face_point is not None and pelvis_center is not None:
                dir_vec  = face_point - pelvis_center
                norm_dir = float(np.linalg.norm(dir_vec))
                if norm_dir > 1e-4:
                    dir_u = dir_vec / norm_dir

        # ====== 中間大圈：最終 output（有投影 & 箭頭） ======

        # 先用 scaled_V 當原始振動
        final_V = scaled_V.copy()

        # 有朝向就做投影：final_V = scaled_V 在 dir_u 上的分量
        if fusion_center is not None and dir_u is not None:
            t = float(np.dot(scaled_V, dir_u))   # 標量投影
            final_V = dir_u * t

        center_output = center_center + final_V

        # 限制在大圈內
        offset_center = center_output - center_center
        mag_center = float(np.linalg.norm(offset_center))
        if mag_center > target_radius and mag_center > 1e-4:
            offset_center *= target_radius / mag_center
            center_output = center_center + offset_center

        center_output[0] = np.clip(center_output[0], 0, w - 1)
        center_output[1] = np.clip(center_output[1], 0, h - 1)

        # 箭頭基底 & 頂端（畫在中間那個圈）
        arrow_base_center = None
        arrow_tip_center  = None
        if dir_u is not None:
            arrow_base_center = center_center - dir_u * target_radius
            arrow_tip_center  = center_center + dir_u * target_radius

        # ====== 右下小圈：原始未投影 output ======
        raw_output = compass_center + scaled_V
        offset_raw = raw_output - compass_center
        mag_raw = float(np.linalg.norm(offset_raw))
        if mag_raw > target_radius and mag_raw > 1e-4:
            offset_raw *= target_radius / mag_raw
            raw_output = compass_center + offset_raw

        raw_output[0] = np.clip(raw_output[0], 0, w - 1)
        raw_output[1] = np.clip(raw_output[1], 0, h - 1)

        # ====== Debug 文字 ======
        y0, dy, line = 15, 14, 0
        if fusion_center is not None:
            txt = f"P_curr: ({fusion_center[0]:.1f}, {fusion_center[1]:.1f})"
            cv2.putText(
                gray_bgr, txt, (5, y0 + line * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255),
                1, cv2.LINE_AA,
            )
            line += 1

        if envelope > 0:
            txt_env = f"E(t): {envelope:.4f}"
            cv2.putText(
                gray_bgr, txt_env, (5, y0 + line * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200),
                1, cv2.LINE_AA,
            )
            line += 1

        if ema_dt is not None and ema_dt > 1e-4:
            fps_text = f"Est. FPS: {SAMPLE_FPS:.2f}"
            cv2.putText(
                gray_bgr, fps_text, (5, y0 + line * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0),
                1, cv2.LINE_AA,
            )

        # ====== 中間大圈 ======
        center_int = (int(center_center[0]), int(center_center[1]))
        cv2.circle(
            gray_bgr,
            center_int,
            int(target_radius),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 中間的朝向箭頭（骨盆→頭，長度 = 圓直徑）
        if arrow_base_center is not None and arrow_tip_center is not None:
            base_pt = (int(arrow_base_center[0]), int(arrow_base_center[1]))
            tip_pt  = (int(arrow_tip_center[0]),  int(arrow_tip_center[1]))
            cv2.arrowedLine(
                gray_bgr,
                base_pt,
                tip_pt,
                (255, 0, 255),
                2,
                tipLength=0.15
            )

        # 中間的「投影後」黃點
        cv2.circle(
            gray_bgr,
            (int(center_output[0]), int(center_output[1])),
            5,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        # ====== 右下小圈（原始 output，不投影，也沒有箭頭） ======
        compass_int = (int(compass_center[0]), int(compass_center[1]))
        cv2.circle(
            gray_bgr,
            compass_int,
            int(target_radius),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.circle(
            gray_bgr,
            (int(raw_output[0]), int(raw_output[1])),
            6,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        with lock:
            processed_frame = gray_bgr.copy()

    pose.close()





# 啟動 worker threads
t1 = threading.Thread(target=capture_loop)
t2 = threading.Thread(target=processing_loop)

t1.start()
t2.start()

# 主執行緒：顯示畫面
try:
    while running:
        frame_to_show = None
        with lock:
            if processed_frame is not None:
                frame_to_show = processed_frame.copy()
            elif latest_frame is not None:
                frame_to_show = latest_frame.copy()

        if frame_to_show is not None:
            cv2.imshow(f"Pose @ {int(TARGET_FPS)} FPS (gray + global freq)", frame_to_show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break
except KeyboardInterrupt:
    print("Ctrl+C pressed — exiting…")
    running = False

t1.join(timeout=1)
t2.join(timeout=1)
cv2.destroyAllWindows()
