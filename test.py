import cv2
import mediapipe as mp
import threading
import time
import numpy as np
from collections import deque

mp_pose = mp.solutions.pose
SMA_WINDOW = 3          # 用於濾波
ENV_WINDOW = 3          # 用於 E(t)，讓大小變化平滑一點

# 全域共享
latest_frame = None        # 最新原始畫面（彩色）
processed_frame = None     # 已畫好 Pose 的畫面（灰階或灰階+骨架）
latest_frame_id = -1       # 目前最新畫面是第幾幀
lock = threading.Lock()
running = True

VIDEO_PATH = "test_video1.MP4"
TARGET_FPS = 10.0          # 播放 & 取樣幀率
SAMPLE_FPS = TARGET_FPS    # 真正給頻率分析用的採樣率（由 processing loop 動態估）

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

    # ====== 共用狀態變數 ======
    last_sample_time = None
    ema_dt = None
    last_processed_id = -1

    # 鏡頭切換閾值：畫面 10%
    cut_thresh = 0.1 * min(SMALL_W, SMALL_H)

    target_radius   = 30.0   # 圓盤半徑 & 動態增益目標振幅
    debug_margin    = 10
    row_spacing     = 2 * target_radius + 20

    SAMPLE_FPS      = TARGET_FPS

    # ====== Mediapipe / Optical-flow 後端狀態 ======
    # MediaPipe
    prev_P_MP   = None
    window_P_MP = deque(maxlen=SMA_WINDOW)
    window_M_MP = deque(maxlen=ENV_WINDOW)
    envelope_MP = 0.0
    axis_MP_last = None

    # Optical Flow
    prev_gray   = None
    prev_pts    = None
    P_virtual   = None
    prev_P_OF   = None
    window_P_OF = deque(maxlen=SMA_WINDOW)
    window_M_OF = deque(maxlen=ENV_WINDOW)
    envelope_OF = 0.0
    axis_OF_last = None

    # goodFeaturesToTrack 參數
    FLOW_MAX_CORNERS = 80
    FLOW_QUALITY     = 0.01
    FLOW_MIN_DIST    = 7

    eps = 1e-4

    def draw_axis_arrow(img, center, axis, radius, color, thickness=2):
        if axis is None:
            return
        axis = np.asarray(axis, dtype=np.float32)
        norm = float(np.linalg.norm(axis))
        if norm < eps:
            return
        u = axis / norm
        base = center - u * radius
        tip  = center + u * radius
        base_pt = (int(base[0]), int(base[1]))
        tip_pt  = (int(tip[0]),  int(tip[1]))
        cv2.arrowedLine(
            img,
            base_pt,
            tip_pt,
            color,
            thickness,
            cv2.LINE_AA,
            tipLength=0.2,
        )

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

        # MediaPipe Pose
        result = pose.process(gray_rgb)

        h, w, _ = gray_bgr.shape

        # 中間大圈圓心（最終 output）
        center_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        # 右上/右下 四顆 debug 球的中心
        mp_row_y = debug_margin + target_radius
        of_row_y = mp_row_y + row_spacing

        mp_proj_center = np.array(
            [w - debug_margin - 3 * target_radius, mp_row_y],
            dtype=np.float32,
        )
        mp_raw_center = np.array(
            [w - debug_margin - 1 * target_radius, mp_row_y],
            dtype=np.float32,
        )
        of_proj_center = np.array(
            [w - debug_margin - 3 * target_radius, of_row_y],
            dtype=np.float32,
        )
        of_raw_center = np.array(
            [w - debug_margin - 1 * target_radius, of_row_y],
            dtype=np.float32,
        )


        # ====== MediaPipe 取得質心 & 軸向 & Guard 資訊 ======
        fusion_center_mp = None   # Raw_Point_MP
        axis_MP = None
        face_point = None
        pelvis_center = None
        avg_visibility = 0.0
        body_size_norm = 0.0

        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            draw_selected_landmarks(gray_bgr, result.pose_landmarks)

            # 14 點平均 = fusion_center_mp
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
                fusion_center_mp = np.array([cx, cy], dtype=np.float32)

            # 臉：鼻子 (0)
            nose_lm = lms[0]
            nx = nose_lm.x * SMALL_W
            ny = nose_lm.y * SMALL_H
            if 0 <= nx < SMALL_W and 0 <= ny < SMALL_H:
                face_point = np.array([nx, ny], dtype=np.float32)

            # 骨盆：左右肩 (11,12) 平均（當作上半身中心）
            lh_lm = lms[11]
            rh_lm = lms[12]
            lx, ly = lh_lm.x * SMALL_W, lh_lm.y * SMALL_H
            rx, ry = rh_lm.x * SMALL_W, rh_lm.y * SMALL_H
            if (0 <= lx < SMALL_W and 0 <= ly < SMALL_H and
                0 <= rx < SMALL_W and 0 <= ry < SMALL_H):
                pelvis_center = np.array([(lx + rx) / 2.0,
                                          (ly + ry) / 2.0],
                                         dtype=np.float32)

            # Guard 用：visibility 平均 & 骨架大小
            vis_list = [lm.visibility for lm in lms]
            avg_visibility = float(np.mean(vis_list))

            xs_n = [lm.x for lm in lms]
            ys_n = [lm.y for lm in lms]
            box_w = max(xs_n) - min(xs_n)
            box_h = max(ys_n) - min(ys_n)
            body_size_norm = (box_w ** 2 + box_h ** 2) ** 0.5  # 對角線長，0~sqrt(2)

            # 當前 Body Axis
            if face_point is not None and pelvis_center is not None:
                dir_vec = face_point - pelvis_center
                norm_dir = float(np.linalg.norm(dir_vec))
                if norm_dir > eps:
                    axis_MP = dir_vec / norm_dir
                    axis_MP_last = axis_MP.copy()

        # Guard：決定最終輸出使用哪一路
        use_mediapipe = False
        GUARD_VIS_THRESH  = 0.7
        GUARD_SIZE_THRESH = 0.45   # 身體至少佔畫面 ~45% 對角線，避免抓到小人偶

        if fusion_center_mp is not None:
            cond_pose = True
            cond_vis  = (avg_visibility > GUARD_VIS_THRESH)
            cond_size = (body_size_norm > GUARD_SIZE_THRESH)
            use_mediapipe = cond_pose and cond_vis and cond_size

        # ====== Optical Flow：計算 v_flow & 更新 P_virtual ======
        if prev_gray is None:
            prev_gray = gray.copy()
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=FLOW_MAX_CORNERS,
                qualityLevel=FLOW_QUALITY,
                minDistance=FLOW_MIN_DIST,
                blockSize=7,
            )

        v_flow = np.zeros(2, dtype=np.float32)

        if prev_gray is not None and prev_pts is not None and len(prev_pts) > 0:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                prev_pts,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )

            if next_pts is not None and status is not None:
                good_new = next_pts[status.reshape(-1) == 1]
                good_old = prev_pts[status.reshape(-1) == 1]

                if len(good_new) > 0:
                    diffs = good_new - good_old
                    mean_diff = np.mean(diffs, axis=0)
                    v_flow = mean_diff.astype(np.float32)

        # ---- 更新 P_virtual（光流虛擬質心） ----
        if P_virtual is None:
            # 第一次就從畫面中心開始
            P_virtual = center_center.copy()
        else:
            # 確保是 1D numpy 向量
            P_virtual = np.asarray(P_virtual, dtype=np.float32).reshape(-1)
            # 如果形狀怪怪的就重設
            if P_virtual.shape[0] != 2:
                P_virtual = center_center.copy()

        # v_flow 也確保成 2D 向量
        v_flow = np.asarray(v_flow, dtype=np.float32).reshape(-1)
        if v_flow.shape[0] != 2:
            v_flow = np.zeros(2, dtype=np.float32)

        # 累加成虛擬位置
        P_virtual = P_virtual + v_flow

        # clamp 在畫面內（不用再包 float，直接賦值就好）
        P_virtual[0] = np.clip(P_virtual[0], 0.0, SMALL_W  - 1.0)
        P_virtual[1] = np.clip(P_virtual[1], 0.0, SMALL_H - 1.0)



        # ====== FPS 估計（以 processing loop 節奏為主） ======
        now = time.time()
        if last_sample_time is not None:
            dt = now - last_sample_time
            if dt > eps:
                if ema_dt is None:
                    ema_dt = dt
                else:
                    ema_dt = 0.9 * ema_dt + 0.1 * dt
        last_sample_time = now

        if ema_dt is not None and ema_dt > eps:
            SAMPLE_FPS = 1.0 / ema_dt
        else:
            SAMPLE_FPS = TARGET_FPS

        # ====== 後端共用：SMA → Project → WindowMax → Gain ======

        # 初始化五顆球的 offset & 軸
        offset_MP_proj = np.zeros(2, dtype=np.float32)
        offset_MP_raw  = np.zeros(2, dtype=np.float32)
        offset_OF_proj = np.zeros(2, dtype=np.float32)
        offset_OF_raw  = np.zeros(2, dtype=np.float32)
        axis_MP_used   = None
        axis_OF_used   = None

        # ---------- MediaPipe 路徑 ----------
        if fusion_center_mp is not None:
            P_mp = fusion_center_mp

            # Cut detection
            if prev_P_MP is not None:
                dist = float(np.linalg.norm(P_mp - prev_P_MP))
                if dist > cut_thresh:
                    window_P_MP.clear()
                    window_M_MP.clear()
                    envelope_MP = 0.0
            prev_P_MP = P_mp.copy()

            # 1. SMA
            window_P_MP.append(P_mp.copy())
            if len(window_P_MP) > 0:
                baseline_MP = np.mean(window_P_MP, axis=0)
            else:
                baseline_MP = P_mp.copy()
            V_MP = P_mp - baseline_MP   # 2D 震動向量

            # 2. Project → 1D
            if axis_MP is not None:
                axis_MP_used = axis_MP
            elif axis_MP_last is not None:
                axis_MP_used = axis_MP_last.copy()
            else:
                axis_MP_used = None

            v1d_MP = 0.0
            if axis_MP_used is not None:
                v1d_MP = float(np.dot(V_MP, axis_MP_used))

            # 3. WindowMax on |v1d|
            M_inst_MP = abs(v1d_MP)
            window_M_MP.append(M_inst_MP)
            envelope_MP = max(window_M_MP) if len(window_M_MP) > 0 else 0.0

            # 4. Dynamic Gain
            if envelope_MP < eps:
                gain_MP = 0.0
                v1d_MP_scaled = 0.0
            else:
                gain_MP = target_radius / envelope_MP
                v1d_MP_scaled = v1d_MP * gain_MP

            if axis_MP_used is not None:
                offset_MP_proj = axis_MP_used * v1d_MP_scaled  # 投影後 2D
            else:
                offset_MP_proj = np.zeros(2, dtype=np.float32)

            offset_MP_raw = V_MP * gain_MP

            # 限制 raw 在圓內
            norm_raw = float(np.linalg.norm(offset_MP_raw))
            if norm_raw > target_radius and norm_raw > eps:
                offset_MP_raw *= target_radius / norm_raw

        # ---------- Optical Flow 路徑 ----------
        if P_virtual is not None:
            P_of = P_virtual

            if prev_P_OF is not None:
                dist = float(np.linalg.norm(P_of - prev_P_OF))
                if dist > cut_thresh:
                    window_P_OF.clear()
                    window_M_OF.clear()
                    envelope_OF = 0.0
            prev_P_OF = P_of.copy()

            # 1. SMA
            window_P_OF.append(P_of.copy())
            if len(window_P_OF) > 0:
                baseline_OF = np.mean(window_P_OF, axis=0)
            else:
                baseline_OF = P_of.copy()
            V_OF = P_of - baseline_OF   # 2D 震動向量

            # 2. Project：軸向取當下 or last_dir
            axis_OF = None
            norm_V_of = float(np.linalg.norm(V_OF))
            if norm_V_of > eps:
                axis_OF = V_OF / norm_V_of   # 即時震動方向
                axis_OF_last = axis_OF.copy()
            elif axis_OF_last is not None:
                axis_OF = axis_OF_last.copy()

            axis_OF_used = axis_OF

            v1d_OF = 0.0
            if axis_OF_used is not None:
                v1d_OF = float(np.dot(V_OF, axis_OF_used))  # = norm(V_OF) 如果 axis=V/||V||

            # 3. WindowMax
            M_inst_OF = abs(v1d_OF)
            window_M_OF.append(M_inst_OF)
            envelope_OF = max(window_M_OF) if len(window_M_OF) > 0 else 0.0

            # 4. Dynamic Gain
            if envelope_OF < eps:
                gain_OF = 0.0
                v1d_OF_scaled = 0.0
            else:
                gain_OF = target_radius / envelope_OF
                v1d_OF_scaled = v1d_OF * gain_OF

            if axis_OF_used is not None:
                offset_OF_proj = axis_OF_used * v1d_OF_scaled
            else:
                offset_OF_proj = np.zeros(2, dtype=np.float32)

            offset_OF_raw = V_OF * gain_OF
            norm_raw_of = float(np.linalg.norm(offset_OF_raw))
            if norm_raw_of > target_radius and norm_raw_of > eps:
                offset_OF_raw *= target_radius / norm_raw_of

        # ====== 最終輸出球：根據 Guard 選其中一路 ======
        if use_mediapipe:
            final_offset = offset_MP_proj
            final_axis   = axis_MP_used
        else:
            final_offset = offset_OF_proj
            final_axis   = None #axis_OF_used

        center_output = center_center + final_offset

        # 保險：限制 final 在大圈內
        offset_center = center_output - center_center
        mag_center = float(np.linalg.norm(offset_center))
        if mag_center > target_radius and mag_center > eps:
            offset_center *= target_radius / mag_center
            center_output = center_center + offset_center

        center_output[0] = np.clip(center_output[0], 0, w - 1)
        center_output[1] = np.clip(center_output[1], 0, h - 1)

        # ====== Debug 文字 ======
        y0, dy, line = 15, 14, 0

        mode_str = "MP" if use_mediapipe else "FLOW"
        cv2.putText(
            gray_bgr, f"Mode: {mode_str}", (5, y0 + line * dy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255),
            1, cv2.LINE_AA,
        )
        line += 1

        fps_text = f"Est. FPS: {SAMPLE_FPS:.2f}"
        cv2.putText(
            gray_bgr, fps_text, (5, y0 + line * dy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0),
            1, cv2.LINE_AA,
        )
        line += 1

        txt_mp_env = f"MP env: {envelope_MP:.3f}"
        cv2.putText(
            gray_bgr, txt_mp_env, (5, y0 + line * dy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200),
            1, cv2.LINE_AA,
        )
        line += 1

        txt_of_env = f"OF env: {envelope_OF:.3f}"
        cv2.putText(
            gray_bgr, txt_of_env, (5, y0 + line * dy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255),
            1, cv2.LINE_AA,
        )

        # ====== 畫三組大圈 / 小圈 ======

        # 1. 中間大圈（最終輸出）
        center_int = (int(center_center[0]), int(center_center[1]))
        cv2.circle(
            gray_bgr,
            center_int,
            int(target_radius),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        draw_axis_arrow(gray_bgr, center_center, final_axis, target_radius, (255, 0, 255))

        cv2.circle(
            gray_bgr,
            (int(center_output[0]), int(center_output[1])),
            5,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        # 2. 右上兩顆：MediaPipe
        mp_proj_int = (int(mp_proj_center[0]), int(mp_proj_center[1]))
        mp_raw_int  = (int(mp_raw_center[0]),  int(mp_raw_center[1]))

        cv2.circle(
            gray_bgr,
            mp_proj_int,
            int(target_radius),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(
            gray_bgr,
            mp_raw_int,
            int(target_radius),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # MediaPipe 投影球箭頭 + 點
        draw_axis_arrow(gray_bgr, mp_proj_center, axis_MP_used, target_radius, (0, 0, 255))
        mp_proj_dot = mp_proj_center + offset_MP_proj
        mp_proj_dot = np.clip(mp_proj_dot, [0, 0], [w - 1, h - 1])
        cv2.circle(
            gray_bgr,
            (int(mp_proj_dot[0]), int(mp_proj_dot[1])),
            4,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        mp_raw_dot = mp_raw_center + offset_MP_raw
        mp_raw_dot = np.clip(mp_raw_dot, [0, 0], [w - 1, h - 1])
        cv2.circle(
            gray_bgr,
            (int(mp_raw_dot[0]), int(mp_raw_dot[1])),
            4,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        # 3. 右下兩顆：Optical Flow
        of_proj_int = (int(of_proj_center[0]), int(of_proj_center[1]))
        of_raw_int  = (int(of_raw_center[0]),  int(of_raw_center[1]))

        cv2.circle(
            gray_bgr,
            of_proj_int,
            int(target_radius),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(
            gray_bgr,
            of_raw_int,
            int(target_radius),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        draw_axis_arrow(gray_bgr, of_proj_center, axis_OF_used, target_radius, (0, 0, 255))
        of_proj_dot = of_proj_center + offset_OF_proj
        of_proj_dot = np.clip(of_proj_dot, [0, 0], [w - 1, h - 1])
        cv2.circle(
            gray_bgr,
            (int(of_proj_dot[0]), int(of_proj_dot[1])),
            4,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        of_raw_dot = of_raw_center + offset_OF_raw
        of_raw_dot = np.clip(of_raw_dot, [0, 0], [w - 1, h - 1])
        cv2.circle(
            gray_bgr,
            (int(of_raw_dot[0]), int(of_raw_dot[1])),
            4,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        # 4. 高亮框：用藍色框出目前採用的那一排兩顆球
        highlight_color  = (255, 0, 0)  # BGR → 藍色
        highlight_thick  = 2
        highlight_margin = 6            # 框框比圓稍微大一點

        if use_mediapipe:
            # 框住上排：mp_proj_center + mp_raw_center
            row_y   = mp_row_y
            left_x  = mp_proj_center[0]
            right_x = mp_raw_center[0]
        else:
            # 框住下排：of_proj_center + of_raw_center
            row_y   = of_row_y
            left_x  = of_proj_center[0]
            right_x = of_raw_center[0]

        x_min = min(left_x, right_x) - target_radius - highlight_margin
        x_max = max(left_x, right_x) + target_radius + highlight_margin
        y_min = row_y - target_radius - highlight_margin
        y_max = row_y + target_radius + highlight_margin

        # 保險：clip 在畫面內
        x_min = int(max(0, x_min))
        x_max = int(min(w - 1, x_max))
        y_min = int(max(0, y_min))
        y_max = int(min(h - 1, y_max))

        cv2.rectangle(
            gray_bgr,
            (x_min, y_min),
            (x_max, y_max),
            highlight_color,
            highlight_thick,
            cv2.LINE_AA,
        )


        # ====== 更新光流狀態，準備下一幀 ======
        prev_gray = gray.copy()
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=FLOW_MAX_CORNERS,
            qualityLevel=FLOW_QUALITY,
            minDistance=FLOW_MIN_DIST,
            blockSize=7,
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
            cv2.imshow(f"Pose @ {int(TARGET_FPS)} FPS (gray + dual mode)", frame_to_show)

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
