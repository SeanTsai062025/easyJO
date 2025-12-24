import cv2
import mediapipe as mp
import threading
import time
import numpy as np
from collections import deque

# =========================
# Optional: YOLO (Ultralytics)
# =========================
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

mp_pose = mp.solutions.pose

SMA_WINDOW = 3
ENV_WINDOW = 3

latest_frame = None
processed_frame = None
latest_frame_id = -1
lock = threading.Lock()
running = True

VIDEO_PATH = "test_video1.MP4"
TARGET_FPS = 10.0
SAMPLE_FPS = TARGET_FPS

VISIBLE_POINTS = list(range(1, 13)) + [23, 24]

MIN_FREQ = 1.0
MAX_FREQ = 4.0

def draw_selected_landmarks(image, landmarks):
    h, w, _ = image.shape
    for idx in VISIBLE_POINTS:
        lm = landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

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

    if src_fps > TARGET_FPS:
        frames_to_skip = int(src_fps / TARGET_FPS) - 1
    else:
        frames_to_skip = 0

    frame_interval = 1.0 / TARGET_FPS
    SAMPLE_FPS = TARGET_FPS
    print(f"src_fps = {src_fps:.2f}, SAMPLE_FPS = {SAMPLE_FPS:.2f}")

    frame_id = 0
    last_time = time.time()

    while running:
        ret, frame = cap.read()
        if not ret:
            running = False
            break

        with lock:
            latest_frame = frame.copy()
            latest_frame_id = frame_id

        frame_id += 1

        for _ in range(frames_to_skip):
            ret, _ = cap.read()
            if not ret:
                running = False
                break

        now = time.time()
        elapsed = now - last_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        last_time = time.time()

    cap.release()

# ----------------------------
# 共用：訊號處理（SMA→V→envelope→gain→offset_raw）
# envelope/gain 的 v1d 定義：
# - MP：用 body axis 做 dot（維持你原本數學）
# - 其他：用 ||V||（等價於 axis=V/||V|| 的 dot）
# ----------------------------
eps = 1e-4

class SignalState:
    def __init__(self, sma_w=3, env_w=3):
        self.prev_P = None
        self.window_P = deque(maxlen=sma_w)
        self.window_M = deque(maxlen=env_w)
        self.envelope = 0.0

def update_signal(P, state: SignalState, target_radius: float, cut_thresh: float, v1d_for_gain: float):
    """
    P: np.array([x,y]) or None
    v1d_for_gain: 用來做 envelope/gain 的一維量（不影響顯示，顯示永遠用 raw 2D）
    return: offset_raw(2D), V(2D), gain, envelope, ok
    """
    if P is None:
        return np.zeros(2, np.float32), np.zeros(2, np.float32), 0.0, state.envelope, False

    P = np.asarray(P, dtype=np.float32).reshape(-1)
    if P.shape[0] != 2:
        return np.zeros(2, np.float32), np.zeros(2, np.float32), 0.0, state.envelope, False

    # Cut detection
    if state.prev_P is not None:
        dist = float(np.linalg.norm(P - state.prev_P))
        if dist > cut_thresh:
            state.window_P.clear()
            state.window_M.clear()
            state.envelope = 0.0
    state.prev_P = P.copy()

    # SMA baseline
    state.window_P.append(P.copy())
    baseline = np.mean(state.window_P, axis=0) if len(state.window_P) > 0 else P.copy()
    V = P - baseline  # 2D vibration vector

    # Envelope on |v1d|
    M_inst = float(abs(v1d_for_gain))
    state.window_M.append(M_inst)
    state.envelope = max(state.window_M) if len(state.window_M) > 0 else 0.0

    # Dynamic gain
    if state.envelope < eps:
        gain = 0.0
        offset_raw = np.zeros(2, np.float32)
    else:
        gain = target_radius / state.envelope
        offset_raw = V * gain

    # Clamp inside circle
    mag = float(np.linalg.norm(offset_raw))
    if mag > target_radius and mag > eps:
        offset_raw *= target_radius / mag

    return offset_raw.astype(np.float32), V.astype(np.float32), float(gain), float(state.envelope), True


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

    last_sample_time = None
    ema_dt = None
    last_processed_id = -1

    # 顯示球參數
    target_radius = 30.0
    margin = 12
    label_h = 18

    # Cut detection threshold
    cut_thresh = 0.1 * min(SMALL_W, SMALL_H)

    # 四路訊號狀態
    st_mp   = SignalState(SMA_WINDOW, ENV_WINDOW)
    st_flow = SignalState(SMA_WINDOW, ENV_WINDOW)
    st_csrt = SignalState(SMA_WINDOW, ENV_WINDOW)
    st_yolo = SignalState(SMA_WINDOW, ENV_WINDOW)

    # ========== (2) Global Optical Flow 狀態 ==========
    prev_gray_flow = None
    prev_pts_flow  = None
    P_virtual_flow = None

    FLOW_MAX_CORNERS = 80
    FLOW_QUALITY     = 0.01
    FLOW_MIN_DIST    = 7

    # ========== (3) CSRT Tracker 狀態 ==========
    tracker_csrt = None
    csrt_inited = False
    csrt_bbox = None

    # ========== (4) YOLO-guided Optical Flow 狀態 ==========
    yolo_model = None
    if YOLO_AVAILABLE:
        # 你也可以換 yolov8s.pt / 自己訓練的權重
        yolo_model = YOLO("yolov8n.pt")

    yolo_interval = 3
    yolo_count = 0
    yolo_bbox = None
    prev_gray_yolo = None
    prev_pts_yolo  = None
    P_virtual_yolo = None
    yolo_lost_count = 0
    yolo_lost_max = 10

    def make_csrt():
        # OpenCV 版本不同 API 可能不一樣
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        return None

    def clamp_bbox(b, w, h):
        x, y, bw, bh = b
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))
        bw = max(1, min(int(bw), w - x))
        bh = max(1, min(int(bh), h - y))
        return (x, y, bw, bh)

    def bbox_center(b):
        x, y, bw, bh = b
        return np.array([x + bw / 2.0, y + bh / 2.0], dtype=np.float32)

    while running:
        frame_copy = None
        cur_id = None

        with lock:
            if latest_frame is not None and latest_frame_id != last_processed_id:
                frame_copy = latest_frame.copy()
                cur_id = latest_frame_id

        if frame_copy is None:
            time.sleep(0.001)
            continue

        last_processed_id = cur_id

        small = cv2.resize(frame_copy, (SMALL_W, SMALL_H))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray_rgb = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2RGB)

        h, w, _ = gray_bgr.shape

        # FPS estimate
        now = time.time()
        if last_sample_time is not None:
            dt = now - last_sample_time
            if dt > eps:
                ema_dt = dt if ema_dt is None else (0.9 * ema_dt + 0.1 * dt)
        last_sample_time = now
        SAMPLE_FPS = (1.0 / ema_dt) if (ema_dt is not None and ema_dt > eps) else TARGET_FPS

        # =========================
        # (1) MediaPipe Pose → centroid + axis (只用來算 v1d/gain)
        # =========================
        fusion_center_mp = None
        axis_mp_used = None

        result = pose.process(gray_rgb)
        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            draw_selected_landmarks(gray_bgr, result.pose_landmarks)

            xs, ys = [], []
            for idx in VISIBLE_POINTS:
                lm = lms[idx]
                x_px = lm.x * SMALL_W
                y_px = lm.y * SMALL_H
                if 0 <= x_px < SMALL_W and 0 <= y_px < SMALL_H:
                    xs.append(x_px)
                    ys.append(y_px)
            if xs and ys:
                fusion_center_mp = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float32)

            # body axis：nose(0) - shoulder_center(11,12)  （你原本叫 pelvis_center 但其實是肩中心）
            nose = lms[0]
            lh = lms[11]
            rh = lms[12]
            nx, ny = nose.x * SMALL_W, nose.y * SMALL_H
            lx, ly = lh.x * SMALL_W, lh.y * SMALL_H
            rx, ry = rh.x * SMALL_W, rh.y * SMALL_H
            if (0 <= nx < SMALL_W and 0 <= ny < SMALL_H and
                0 <= lx < SMALL_W and 0 <= ly < SMALL_H and
                0 <= rx < SMALL_W and 0 <= ry < SMALL_H):
                face_pt = np.array([nx, ny], dtype=np.float32)
                shoulder_ctr = np.array([(lx + rx) / 2.0, (ly + ry) / 2.0], dtype=np.float32)
                d = face_pt - shoulder_ctr
                nd = float(np.linalg.norm(d))
                if nd > eps:
                    axis_mp_used = (d / nd).astype(np.float32)

        # =========================
        # (2) Global Optical Flow → P_virtual_flow
        # =========================
        if prev_gray_flow is None:
            prev_gray_flow = gray.copy()
            prev_pts_flow = cv2.goodFeaturesToTrack(
                prev_gray_flow,
                maxCorners=FLOW_MAX_CORNERS,
                qualityLevel=FLOW_QUALITY,
                minDistance=FLOW_MIN_DIST,
                blockSize=7,
            )

        v_flow = np.zeros(2, dtype=np.float32)

        next_pts = None
        status = None

        if prev_gray_flow is not None and prev_pts_flow is not None and len(prev_pts_flow) > 0:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray_flow, gray, prev_pts_flow, None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )

        if next_pts is not None and status is not None:
            good_new = next_pts[status.reshape(-1) == 1]
            good_old = prev_pts_flow[status.reshape(-1) == 1]

            if len(good_new) > 0:
                diffs = (good_new - good_old).reshape(-1, 2)   # 重要：避免 (1,2) shape
                v_flow = diffs.mean(axis=0).astype(np.float32)



        P_virtual_flow = np.asarray(P_virtual_flow, dtype=np.float32).reshape(-1)
        if P_virtual_flow.shape[0] != 2:
            P_virtual_flow = np.array([w/2.0, h/2.0], dtype=np.float32)

        P_virtual_flow = P_virtual_flow + v_flow
        P_virtual_flow[0] = np.clip(P_virtual_flow[0], 0.0, w - 1.0)
        P_virtual_flow[1] = np.clip(P_virtual_flow[1], 0.0, h - 1.0)


        # 更新 global flow 的 prev
        prev_gray_flow = gray.copy()
        prev_pts_flow = cv2.goodFeaturesToTrack(
            prev_gray_flow,
            maxCorners=FLOW_MAX_CORNERS,
            qualityLevel=FLOW_QUALITY,
            minDistance=FLOW_MIN_DIST,
            blockSize=7,
        )

        # =========================
        # (3) CSRT Tracker → bbox center
        # 初始化策略：如果 MP 有偵測到，就用 MP landmarks 的 bbox 來 init tracker
        # =========================
        csrt_center = None

        if not csrt_inited:
            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark
                xs = [lm.x for lm in lms]
                ys = [lm.y for lm in lms]
                x1 = int(max(0, (min(xs) * w) - 10))
                y1 = int(max(0, (min(ys) * h) - 10))
                x2 = int(min(w - 1, (max(xs) * w) + 10))
                y2 = int(min(h - 1, (max(ys) * h) + 10))
                bw = max(20, x2 - x1)
                bh = max(20, y2 - y1)
                csrt_bbox = clamp_bbox((x1, y1, bw, bh), w, h)

                tracker_csrt = make_csrt()
                if tracker_csrt is not None:
                    ok = tracker_csrt.init(gray_bgr, csrt_bbox)
                    csrt_inited = True

        if csrt_inited and tracker_csrt is not None:
            ok, b = tracker_csrt.update(gray_bgr)
            if ok:
                csrt_bbox = clamp_bbox(b, w, h)
                csrt_center = bbox_center(csrt_bbox)
                # 畫 bbox
                x, y, bw, bh = csrt_bbox
                cv2.rectangle(gray_bgr, (x, y), (x + bw, y + bh), (255, 255, 0), 1)
            else:
                # tracker 掛了就重置，等下一次 MP 再 init
                csrt_inited = False
                tracker_csrt = None
                csrt_bbox = None

        # =========================
        # (4) YOLO-guided Optical Flow
        # YOLO 找 person bbox（降低頻率，每 yolo_interval 幀做一次）
        # 光流只在 bbox ROI 內追蹤特徵點
        # =========================
        yolo_center = None

        if YOLO_AVAILABLE and yolo_model is not None:
            yolo_count += 1
            do_yolo = (yolo_bbox is None) or (yolo_count % yolo_interval == 0) or (yolo_lost_count > 0)

            if do_yolo:
                # ultralytics 接收 BGR/np array
                preds = yolo_model.predict(source=small, verbose=False, conf=0.25)
                best = None
                best_conf = -1.0
                if preds and len(preds) > 0:
                    boxes = preds[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            cls = int(box.cls.item())
                            conf = float(box.conf.item())
                            # COCO person = 0
                            if cls == 0 and conf > best_conf:
                                xyxy = box.xyxy[0].cpu().numpy()
                                best = xyxy
                                best_conf = conf

                if best is not None:
                    x1, y1, x2, y2 = best
                    b = (x1, y1, x2 - x1, y2 - y1)
                    yolo_bbox = clamp_bbox(b, w, h)
                    yolo_lost_count = 0
                else:
                    yolo_lost_count += 1
                    if yolo_lost_count > yolo_lost_max:
                        yolo_bbox = None
                        prev_gray_yolo = None
                        prev_pts_yolo = None
                        P_virtual_yolo = None

            if yolo_bbox is not None:
                x, y, bw, bh = yolo_bbox
                cv2.rectangle(gray_bgr, (x, y), (x + bw, y + bh), (0, 165, 255), 1)

                roi_gray = gray[y:y+bh, x:x+bw]
                if roi_gray.size > 0:
                    if prev_gray_yolo is None:
                        prev_gray_yolo = gray.copy()
                        # ROI 內點
                        prev_pts_yolo = cv2.goodFeaturesToTrack(
                            roi_gray,
                            maxCorners=FLOW_MAX_CORNERS,
                            qualityLevel=FLOW_QUALITY,
                            minDistance=FLOW_MIN_DIST,
                            blockSize=7,
                        )
                        if prev_pts_yolo is not None:
                            prev_pts_yolo[:, 0, 0] += x
                            prev_pts_yolo[:, 0, 1] += y
                        P_virtual_yolo = bbox_center(yolo_bbox)
                    else:
                        v_roi = np.zeros(2, dtype=np.float32)
                        if prev_pts_yolo is not None and len(prev_pts_yolo) > 0:
                            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                                prev_gray_yolo, gray, prev_pts_yolo, None,
                                winSize=(21, 21),
                                maxLevel=3,
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                            )
                            if next_pts is not None and status is not None:
                                good_new = next_pts[status.reshape(-1) == 1]
                                good_old = prev_pts_yolo[status.reshape(-1) == 1]
                                if len(good_new) > 0:
                                    diffs = (good_new - good_old).reshape(-1, 2)
                                    v_roi = diffs.mean(axis=0).astype(np.float32)


                        if P_virtual_yolo is None:
                            P_virtual_yolo = bbox_center(yolo_bbox)

                        P_virtual_yolo = (P_virtual_yolo + v_roi).astype(np.float32)
                        P_virtual_yolo[0] = np.clip(P_virtual_yolo[0], x, x + bw - 1)
                        P_virtual_yolo[1] = np.clip(P_virtual_yolo[1], y, y + bh - 1)

                        # refresh points occasionally / if too few
                        need_refresh = (prev_pts_yolo is None) or (len(prev_pts_yolo) < 15)
                        if need_refresh:
                            pts = cv2.goodFeaturesToTrack(
                                roi_gray,
                                maxCorners=FLOW_MAX_CORNERS,
                                qualityLevel=FLOW_QUALITY,
                                minDistance=FLOW_MIN_DIST,
                                blockSize=7,
                            )
                            if pts is not None:
                                pts[:, 0, 0] += x
                                pts[:, 0, 1] += y
                            prev_pts_yolo = pts
                        else:
                            # keep using tracked points
                            prev_pts_yolo = next_pts if 'next_pts' in locals() and next_pts is not None else prev_pts_yolo

                        prev_gray_yolo = gray.copy()

                    yolo_center = P_virtual_yolo.copy()

        # =========================
        # 四路：算 offset_raw（2D）並畫四顆球
        # =========================
        # 先把四個球中心排成 2x2
        # TL: MP, TR: Global Flow, BL: CSRT, BR: YOLO+Flow
        cx1 = margin + target_radius
        cx2 = w - margin - target_radius
        cy1 = margin + label_h + target_radius
        cy2 = h - margin - target_radius

        ball_centers = {
            "1) MP Pose": np.array([cx1, cy1], dtype=np.float32),
            "2) Global Flow": np.array([cx2, cy1], dtype=np.float32),
            "3) CSRT": np.array([cx1, cy2], dtype=np.float32),
            "4) YOLO+Flow": np.array([cx2, cy2], dtype=np.float32),
        }

        # ---------- (1) MP offset_raw ----------
        offset_mp = np.zeros(2, np.float32)
        env_mp = st_mp.envelope
        if fusion_center_mp is not None:
            # 你原本的 v1d = dot(V, body_axis)，我們維持這個用來做 gain
            # 但顯示不用投影：顯示 raw 2D offset
            # 先取得暫時 baseline/V：update_signal 會自己算，但它需要 v1d_for_gain
            # 所以要先做一次「假 v1d」→ 用 prev_P/window_P 算 V 太麻煩
            # 我們用一個小技巧：先用 state.window_P 推 baseline，再算 V，再算 v1d
            # 確保數學跟你原本一樣的順序（baseline在前，v1d基於V）
            # 做法：先預估 baseline
            tmpP = fusion_center_mp
            baseline = np.mean(st_mp.window_P, axis=0) if len(st_mp.window_P) > 0 else tmpP
            V = tmpP - baseline
            v1d = 0.0
            if axis_mp_used is not None:
                v1d = float(np.dot(V, axis_mp_used))

            offset_mp, _, _, env_mp, _ = update_signal(tmpP, st_mp, target_radius, cut_thresh, v1d)

        # ---------- (2) Global Flow offset_raw ----------
        offset_flow = np.zeros(2, np.float32)
        env_flow = st_flow.envelope
        if P_virtual_flow is not None:
            # v1d 用 ||V||（等價於你原本 flow 的 axis=V/||V||）
            baseline = np.mean(st_flow.window_P, axis=0) if len(st_flow.window_P) > 0 else P_virtual_flow
            V = P_virtual_flow - baseline
            v1d = float(np.linalg.norm(V))
            offset_flow, _, _, env_flow, _ = update_signal(P_virtual_flow, st_flow, target_radius, cut_thresh, v1d)

        # ---------- (3) CSRT offset_raw ----------
        offset_csrt = np.zeros(2, np.float32)
        env_csrt = st_csrt.envelope
        if csrt_center is not None:
            baseline = np.mean(st_csrt.window_P, axis=0) if len(st_csrt.window_P) > 0 else csrt_center
            V = csrt_center - baseline
            v1d = float(np.linalg.norm(V))
            offset_csrt, _, _, env_csrt, _ = update_signal(csrt_center, st_csrt, target_radius, cut_thresh, v1d)

        # ---------- (4) YOLO+Flow offset_raw ----------
        offset_yolo = np.zeros(2, np.float32)
        env_yolo = st_yolo.envelope
        if yolo_center is not None:
            baseline = np.mean(st_yolo.window_P, axis=0) if len(st_yolo.window_P) > 0 else yolo_center
            V = yolo_center - baseline
            v1d = float(np.linalg.norm(V))
            offset_yolo, _, _, env_yolo, _ = update_signal(yolo_center, st_yolo, target_radius, cut_thresh, v1d)

        offsets = {
            "1) MP Pose": offset_mp,
            "2) Global Flow": offset_flow,
            "3) CSRT": offset_csrt,
            "4) YOLO+Flow": offset_yolo,
        }
        envs = {
            "1) MP Pose": env_mp,
            "2) Global Flow": env_flow,
            "3) CSRT": env_csrt,
            "4) YOLO+Flow": env_yolo,
        }

        # 畫四顆球 + label
        for name, ctr in ball_centers.items():
            ctr_i = (int(ctr[0]), int(ctr[1]))
            cv2.circle(gray_bgr, ctr_i, int(target_radius), (255, 255, 255), 1, cv2.LINE_AA)

            dot = ctr + offsets[name]
            dot[0] = np.clip(dot[0], 0, w - 1)
            dot[1] = np.clip(dot[1], 0, h - 1)
            cv2.circle(gray_bgr, (int(dot[0]), int(dot[1])), 5, (0, 255, 255), -1, cv2.LINE_AA)

            # label 放在球上方
            label_pos = (int(ctr[0] - target_radius), int(ctr[1] - target_radius - 6))
            cv2.putText(gray_bgr, name, label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1, cv2.LINE_AA)

            # env 顯示（可刪）
            env_pos = (int(ctr[0] - target_radius), int(ctr[1] + target_radius + 14))
            cv2.putText(gray_bgr, f"env:{envs[name]:.3f}", env_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # 角落資訊
        cv2.putText(gray_bgr, f"Est.FPS:{SAMPLE_FPS:.2f}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(gray_bgr, f"YOLO:{'ON' if YOLO_AVAILABLE else 'OFF'}",
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        with lock:
            processed_frame = gray_bgr.copy()

    pose.close()

# threads
t1 = threading.Thread(target=capture_loop)
t2 = threading.Thread(target=processing_loop)
t1.start()
t2.start()

try:
    while running:
        frame_to_show = None
        with lock:
            if processed_frame is not None:
                frame_to_show = processed_frame.copy()
            elif latest_frame is not None:
                frame_to_show = latest_frame.copy()

        if frame_to_show is not None:
            cv2.imshow(f"4-method test @ {int(TARGET_FPS)} FPS", frame_to_show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break
except KeyboardInterrupt:
    running = False

t1.join(timeout=1)
t2.join(timeout=1)
cv2.destroyAllWindows()
