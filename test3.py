import cv2
import mediapipe as mp
import threading
import time
import numpy as np
from collections import deque

mp_pose = mp.solutions.pose

# =========================
# tunable parameters
# =========================
VIDEO_PATH = "test_video/wholevideo.MP4"
TARGET_FPS = 10.0

SMA_WINDOW = 3          # baseline smoothing window (V=P-B)
ENV_WINDOW = 3          # envelope window (amplitude)

# ---- Check A: MP trustworthy ----
MP_VIS_THRESH = 0.7
MIN_SHOULDER_WIDTH_PX = 35
MIN_HIP_WIDTH_PX      = 30
MP_CHECKA_MIN_POINTS = 12

# ---- Check B: Corner-based camera motion detection ----
CORNER_RATIO      = 0.15
CORNER_MOVE_PX    = 0.9
CORNER_MIN_POINTS = 12

# ---- Mode hysteresis ----
SWITCH_TO_MP_N = 1
SWITCH_TO_OF_M = 5

# ---- UI ----
MP_UI_MIN_POINTS = 4
TARGET_RADIUS = 30.0

VISIBLE_POINTS = list(range(1, 13)) + [23, 24]   # 14 points
# =========================

# ---- Debug: 顯示 Optical Flow 追蹤點 ----
DRAW_OF_FULL_TRACKS = True     # 顯示全畫面 OF 追蹤點
DRAW_CORNER_TRACKS  = True     # 顯示 CornerCam 追蹤點
DRAW_TRACK_LINES    = True     # 是否畫 old->new 線
TRACK_DOT_R         = 2        # 點半徑
TRACK_LINE_THICK    = 1
TRACK_MAX_DRAW      = 80       # 最多畫幾個（避免畫面太亂）



# =========================
# Global shared buffers
# =========================
latest_frame = None
processed_frame = None
latest_frame_id = -1
lock = threading.Lock()
running = True


def draw_selected_landmarks(image, landmarks):
    h, w, _ = image.shape
    for idx in VISIBLE_POINTS:
        lm = landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)


def draw_axis_arrow(img, center, axis, radius, color, thickness=2):
    if axis is None:
        return
    axis = np.asarray(axis, dtype=np.float32).reshape(-1)
    if axis.shape[0] != 2:
        return
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return
    u = axis / norm
    base = center - u * radius
    tip  = center + u * radius
    cv2.arrowedLine(
        img,
        (int(base[0]), int(base[1])),
        (int(tip[0]),  int(tip[1])),
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.2,
    )

def draw_lk_tracks(img, good_old, good_new,
                   color_pt=(0,255,0), color_ln=(0,128,255),
                   r=2, thick=1, max_draw=80, draw_lines=True):
    if good_old is None or good_new is None:
        return

    p0 = np.asarray(good_old).reshape(-1, 2)
    p1 = np.asarray(good_new).reshape(-1, 2)
    n = min(len(p0), len(p1), int(max_draw))

    for i in range(n):
        x0, y0 = int(p0[i,0]), int(p0[i,1])
        x1, y1 = int(p1[i,0]), int(p1[i,1])

        if draw_lines:
            cv2.line(img, (x0, y0), (x1, y1),
                     color_ln, thick, cv2.LINE_AA)

        cv2.circle(img, (x1, y1),
                   r, color_pt, -1, cv2.LINE_AA)


def draw_link_arrow(img, p0, p1, radius, color=(255, 0, 0), thickness=2):
    """
    Draw an arrow from circle at p0 to circle at p1.
    p0,p1: (2,) float centers
    radius: circle radius
    """
    p0 = np.asarray(p0, dtype=np.float32).reshape(2)
    p1 = np.asarray(p1, dtype=np.float32).reshape(2)

    d = p1 - p0
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return
    u = d / n

    # start/end slightly outside circle boundary
    start = p0 + u * (radius + 4.0)
    end   = p1 - u * (radius + 8.0)

    cv2.arrowedLine(
        img,
        (int(start[0]), int(start[1])),
        (int(end[0]),   int(end[1])),
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.25,
    )



def capture_loop():
    global latest_frame, running, latest_frame_id
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video")
        running = False
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = TARGET_FPS

    frames_to_skip = 0
    if src_fps > TARGET_FPS:
        frames_to_skip = int(src_fps / TARGET_FPS) - 1
        frames_to_skip = max(0, frames_to_skip)

    frame_interval = 1.0 / TARGET_FPS
    print(f"src_fps = {src_fps:.2f}, TARGET_FPS = {TARGET_FPS:.2f}")

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


def processing_loop():
    global latest_frame, processed_frame, running, latest_frame_id

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    SMALL_W, SMALL_H = 480, 270
    eps = 1e-6

    last_sample_time = None
    ema_dt = None
    sample_fps = TARGET_FPS

    last_processed_id = -1

    # ⚠️ 原本 cut_thresh 太小很容易一直 reset，PCA 永遠不 ready
    cut_thresh = 0.25 * min(SMALL_W, SMALL_H)

    # MP pipeline buffers
    window_P_MP = deque(maxlen=SMA_WINDOW)
    window_M_MP = deque(maxlen=ENV_WINDOW)
    envelope_MP = 0.0
    prev_P_MP = None

    # OF output buffers
    window_P_OF = deque(maxlen=SMA_WINDOW)
    window_M_OF = deque(maxlen=ENV_WINDOW)
    envelope_OF = 0.0
    prev_P_virtual = None

    # OF states
    prev_gray_cam = None
    prev_pts_cam  = None

    prev_gray_full = None
    prev_pts_full  = None
    P_virtual      = None

    axis_MP_last = None

    mp_ui_offset_proj_last = np.zeros(2, dtype=np.float32)
    mp_ui_offset_raw_last  = np.zeros(2, dtype=np.float32)

    mode_use_mp = False
    mp_good_cnt = 0
    mp_bad_cnt  = 0

    FLOW_MAX_CORNERS = 80
    FLOW_QUALITY     = 0.01
    FLOW_MIN_DIST    = 7

    debug_margin = 10
    target_radius = float(TARGET_RADIUS)
    row_spacing = 2 * target_radius + 20
    mp_row_y = debug_margin + target_radius
    of_row_y = mp_row_y + row_spacing

    # ===== PCA axis from V_OF =====
    AXIS_WIN = 20
    AXIS_MIN_SAMPLES = 8
    AXIS_MIN_ENERGY = 1e-3
    AXIS_EMA_ALPHA = 0.25

    window_V_OF = deque(maxlen=AXIS_WIN)
    axis_VOF_last = None

    def maybe_reset_on_cut(curP, prevP, windowP, windowM):
        if prevP is None:
            return False
        dist = float(np.linalg.norm(curP - prevP))
        if dist > cut_thresh:
            windowP.clear()
            windowM.clear()
            return True
        return False

    def estimate_axis_from_V(windowV, prev_axis=None):
        if len(windowV) < AXIS_MIN_SAMPLES:
            return None
        X = np.array(windowV, dtype=np.float32)  # (N,2)
        energy = float(np.mean(np.sum(X * X, axis=1)))
        if energy < AXIS_MIN_ENERGY:
            return None

        Xm = X - np.mean(X, axis=0, keepdims=True)
        C = (Xm.T @ Xm) / max(1, Xm.shape[0])

        evals, evecs = np.linalg.eigh(C)
        axis = evecs[:, int(np.argmax(evals))].astype(np.float32)

        n = float(np.linalg.norm(axis))
        if n < 1e-6:
            return None
        axis /= n

        if prev_axis is not None and float(np.dot(axis, prev_axis)) < 0:
            axis = -axis
        return axis

    # 你已經在外面定義好了 draw_link_arrow / draw_axis_arrow
    # 這裡只用它們，不重複定義

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
        center_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        # Corner ROIs
        roi_w = int(w * CORNER_RATIO)
        roi_h = int(h * CORNER_RATIO)
        ROIS = [
            (0, 0, roi_w, roi_h),
            (w - roi_w, 0, w, roi_h),
            (0, h - roi_h, roi_w, h),
            (w - roi_w, h - roi_h, w, h),
        ]
        corner_mask = np.zeros_like(gray, dtype=np.uint8)
        for (x0, y0, x1, y1) in ROIS:
            corner_mask[y0:y1, x0:x1] = 255

        # =========================
        # MediaPipe (always running)
        # =========================
        result = pose.process(gray_rgb)

        fusion_center_mp_ui = None
        fusion_center_mp_check = None
        mp_avg_vis = 0.0
        mp_valid_points = 0
        shoulder_width_px = 0.0
        hip_width_px      = 0.0

        axis_MP = None
        nose_pt = None
        torso_center = None

        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            draw_selected_landmarks(gray_bgr, result.pose_landmarks)

            xs_in, ys_in, vis_in = [], [], []
            for idx in VISIBLE_POINTS:
                lm = lms[idx]
                x_px = lm.x * SMALL_W
                y_px = lm.y * SMALL_H
                v = float(lm.visibility)
                if 0 <= x_px < SMALL_W and 0 <= y_px < SMALL_H:
                    xs_in.append(float(x_px))
                    ys_in.append(float(y_px))
                    vis_in.append(v)

            mp_valid_points = len(xs_in)

            if mp_valid_points >= MP_UI_MIN_POINTS:
                fusion_center_mp_ui = np.array([float(np.mean(xs_in)), float(np.mean(ys_in))], dtype=np.float32)

            if mp_valid_points >= MP_CHECKA_MIN_POINTS and fusion_center_mp_ui is not None:
                mp_avg_vis = float(np.mean(vis_in))
                fusion_center_mp_check = fusion_center_mp_ui

                ls = lms[11]; rs = lms[12]
                lh = lms[23]; rh = lms[24]
                ls_xy = np.array([ls.x * SMALL_W, ls.y * SMALL_H], dtype=np.float32)
                rs_xy = np.array([rs.x * SMALL_W, rs.y * SMALL_H], dtype=np.float32)
                lh_xy = np.array([lh.x * SMALL_W, lh.y * SMALL_H], dtype=np.float32)
                rh_xy = np.array([rh.x * SMALL_W, rh.y * SMALL_H], dtype=np.float32)
                shoulder_width_px = float(np.linalg.norm(ls_xy - rs_xy))
                hip_width_px      = float(np.linalg.norm(lh_xy - rh_xy))

            nose = lms[0]
            nx, ny = nose.x * SMALL_W, nose.y * SMALL_H
            if 0 <= nx < SMALL_W and 0 <= ny < SMALL_H:
                nose_pt = np.array([nx, ny], dtype=np.float32)

            lh = lms[11]; rh = lms[12]
            lx, ly = lh.x * SMALL_W, lh.y * SMALL_H
            rx, ry = rh.x * SMALL_W, rh.y * SMALL_H
            if (0 <= lx < SMALL_W and 0 <= ly < SMALL_H and 0 <= rx < SMALL_W and 0 <= ry < SMALL_H):
                torso_center = np.array([(lx + rx) / 2.0, (ly + ry) / 2.0], dtype=np.float32)

            if nose_pt is not None and torso_center is not None:
                d = nose_pt - torso_center
                nd = float(np.linalg.norm(d))
                if nd > eps:
                    axis_MP = (d / nd).astype(np.float32)
                    axis_MP_last = axis_MP.copy()

        axis_from_mp = axis_MP if axis_MP is not None else (axis_MP_last.copy() if axis_MP_last is not None else None)

        # =========================
        # CheckB: CornerCam OF (always running)
        # =========================
        is_camera_moving = False
        corner_mag = 0.0
        corner_n = 0

        if prev_gray_cam is None:
            prev_gray_cam = gray.copy()
            prev_pts_cam = cv2.goodFeaturesToTrack(
                prev_gray_cam, maxCorners=FLOW_MAX_CORNERS,
                qualityLevel=FLOW_QUALITY, minDistance=FLOW_MIN_DIST,
                blockSize=7, mask=corner_mask
            )
        else:
            if prev_pts_cam is not None and len(prev_pts_cam) > 0:
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray_cam, gray, prev_pts_cam, None,
                    winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )
                if next_pts is not None and status is not None:
                    good_new = next_pts[status.reshape(-1) == 1]
                    good_old = prev_pts_cam[status.reshape(-1) == 1]

                    if DRAW_CORNER_TRACKS:
                        draw_lk_tracks(
                            gray_bgr,
                            good_old,
                            good_new,
                            color_pt=(0, 255, 255),   # 黃色點
                            color_ln=(255, 255, 0),   # 青色線
                            r=TRACK_DOT_R,
                            thick=TRACK_LINE_THICK,
                            max_draw=TRACK_MAX_DRAW,
                            draw_lines=DRAW_TRACK_LINES
                        )


                    corner_n = len(good_new)
                    if corner_n >= CORNER_MIN_POINTS:
                        diffs = (good_new - good_old).reshape(-1, 2).astype(np.float32)
                        v_corner = np.median(diffs, axis=0).astype(np.float32)
                        corner_mag = float(np.linalg.norm(v_corner))
                        is_camera_moving = (corner_mag > CORNER_MOVE_PX)

            prev_gray_cam = gray.copy()
            prev_pts_cam = cv2.goodFeaturesToTrack(
                prev_gray_cam, maxCorners=FLOW_MAX_CORNERS,
                qualityLevel=FLOW_QUALITY, minDistance=FLOW_MIN_DIST,
                blockSize=7, mask=corner_mask
            )

        # =========================
        # OF_full (always running)
        # =========================
        v_flow_full = np.zeros(2, dtype=np.float32)

        if prev_gray_full is None:
            prev_gray_full = gray.copy()
            prev_pts_full = cv2.goodFeaturesToTrack(
                prev_gray_full, maxCorners=FLOW_MAX_CORNERS,
                qualityLevel=FLOW_QUALITY, minDistance=FLOW_MIN_DIST,
                blockSize=7, mask=None
            )
        else:
            if prev_pts_full is not None and len(prev_pts_full) > 0:
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray_full, gray, prev_pts_full, None,
                    winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )
                if next_pts is not None and status is not None:
                    good_new = next_pts[status.reshape(-1) == 1]
                    good_old = prev_pts_full[status.reshape(-1) == 1]

                    # ---- 顯示 全畫面 Optical Flow 追蹤點 ----
                    if DRAW_OF_FULL_TRACKS:
                        draw_lk_tracks(
                            gray_bgr,
                            good_old,
                            good_new,
                            color_pt=(255, 100, 0),
                            color_ln=(255, 180, 80),
                            r=TRACK_DOT_R,
                            thick=TRACK_LINE_THICK,
                            max_draw=TRACK_MAX_DRAW,
                            draw_lines=DRAW_TRACK_LINES
                        )


                    
                    if len(good_new) > 0:
                        diffs_full = (good_new - good_old).reshape(-1, 2).astype(np.float32)
                        v_flow_full = np.median(diffs_full, axis=0).astype(np.float32)

            prev_gray_full = gray.copy()
            prev_pts_full = cv2.goodFeaturesToTrack(
                prev_gray_full, maxCorners=FLOW_MAX_CORNERS,
                qualityLevel=FLOW_QUALITY, minDistance=FLOW_MIN_DIST,
                blockSize=7, mask=None
            )

        if P_virtual is None:
            P_virtual = center_center.copy()
        else:
            P_virtual = np.asarray(P_virtual, dtype=np.float32).reshape(-1)
            if P_virtual.shape[0] != 2:
                P_virtual = center_center.copy()

        P_virtual = P_virtual + v_flow_full
        P_virtual[0] = np.clip(P_virtual[0], 0.0, SMALL_W - 1.0)
        P_virtual[1] = np.clip(P_virtual[1], 0.0, SMALL_H - 1.0)

        # FPS estimate
        now = time.time()
        if last_sample_time is not None:
            dt = now - last_sample_time
            if dt > eps:
                ema_dt = dt if ema_dt is None else (0.9 * ema_dt + 0.1 * dt)
        last_sample_time = now
        sample_fps = (1.0 / ema_dt) if (ema_dt is not None and ema_dt > eps) else TARGET_FPS

        # =========================
        # CheckA: MP trustworthy
        # =========================
        checkA_mp_trustworthy = False
        if fusion_center_mp_check is not None:
            cond_vis = (mp_avg_vis > MP_VIS_THRESH)
            cond_not_collapsed = (shoulder_width_px >= MIN_SHOULDER_WIDTH_PX) and (hip_width_px >= MIN_HIP_WIDTH_PX)
            checkA_mp_trustworthy = (cond_vis and cond_not_collapsed)

        # =========================
        # Mode decision (your table)
        # Only TT => MP mode, else OF mode
        # =========================
        mp_ok_now = (checkA_mp_trustworthy and is_camera_moving)

        if mp_ok_now:
            mp_good_cnt += 1
            mp_bad_cnt = 0
        else:
            mp_bad_cnt += 1
            mp_good_cnt = 0

        if (not mode_use_mp) and (mp_good_cnt >= SWITCH_TO_MP_N):
            mode_use_mp = True
        if mode_use_mp and (mp_bad_cnt >= SWITCH_TO_OF_M):
            mode_use_mp = False

        use_mediapipe = mode_use_mp
        mode_str = "MP" if use_mediapipe else "FLOW"

        # =========================
        # MP offsets (projected)
        # =========================
        offset_MP_proj = mp_ui_offset_proj_last.copy()
        offset_MP_raw  = mp_ui_offset_raw_last.copy()
        axis_MP_used   = axis_from_mp

        if fusion_center_mp_ui is not None:
            P_mp = fusion_center_mp_ui.astype(np.float32)
            reset = maybe_reset_on_cut(P_mp, prev_P_MP, window_P_MP, window_M_MP)
            if reset:
                envelope_MP = 0.0
            prev_P_MP = P_mp.copy()

            window_P_MP.append(P_mp.copy())
            baseline_MP = np.mean(np.array(window_P_MP), axis=0).astype(np.float32)
            V_MP = (P_mp - baseline_MP).astype(np.float32)

            v1d_MP = float(np.dot(V_MP, axis_MP_used)) if axis_MP_used is not None else 0.0
            window_M_MP.append(abs(v1d_MP))
            envelope_MP = max(window_M_MP) if len(window_M_MP) > 0 else 0.0

            gain_mp = (target_radius / envelope_MP) if envelope_MP > eps else 0.0
            v1d_scaled = v1d_MP * gain_mp

            offset_MP_proj = axis_MP_used * v1d_scaled if axis_MP_used is not None else np.zeros(2, np.float32)
            offset_MP_raw = V_MP * gain_mp

            nraw = float(np.linalg.norm(offset_MP_raw))
            if nraw > target_radius and nraw > eps:
                offset_MP_raw *= target_radius / nraw

            mp_ui_offset_proj_last = offset_MP_proj.copy()
            mp_ui_offset_raw_last  = offset_MP_raw.copy()

        # =========================
        # OF pipeline: update PCA axis first, then select axis_OF_used
        # =========================
        offset_OF_proj = np.zeros(2, dtype=np.float32)
        offset_OF_raw  = np.zeros(2, dtype=np.float32)

        P_of = P_virtual.astype(np.float32)

        if prev_P_virtual is not None:
            dist = float(np.linalg.norm(P_of - prev_P_virtual))
            if dist > cut_thresh:
                window_P_OF.clear()
                window_M_OF.clear()
                envelope_OF = 0.0
                window_V_OF.clear()
        prev_P_virtual = P_of.copy()

        window_P_OF.append(P_of.copy())
        baseline_OF = np.mean(np.array(window_P_OF), axis=0).astype(np.float32)
        V_OF = (P_of - baseline_OF).astype(np.float32)

        # update PCA axis
        window_V_OF.append(V_OF.copy())
        axis_candidate = estimate_axis_from_V(window_V_OF, prev_axis=axis_VOF_last)
        if axis_candidate is not None:
            if axis_VOF_last is None:
                axis_VOF_last = axis_candidate
            else:
                axis_sm = (1.0 - AXIS_EMA_ALPHA) * axis_VOF_last + AXIS_EMA_ALPHA * axis_candidate
                nn = float(np.linalg.norm(axis_sm))
                if nn > 1e-6:
                    axis_VOF_last = (axis_sm / nn).astype(np.float32)

        # select OF axis (your table)
        if checkA_mp_trustworthy:
            axis_OF_used = axis_from_mp
        else:
            axis_OF_used = axis_VOF_last  # may be None if not ready

        # envelope + gain
        if axis_OF_used is not None:
            v1d_OF = float(np.dot(V_OF, axis_OF_used))
            scalar_for_env = abs(v1d_OF)
        else:
            v1d_OF = 0.0
            scalar_for_env = float(np.linalg.norm(V_OF))

        window_M_OF.append(scalar_for_env)
        envelope_OF = max(window_M_OF) if len(window_M_OF) > 0 else 0.0
        gain = (target_radius / envelope_OF) if envelope_OF > eps else 0.0

        offset_OF_raw = V_OF * gain
        nraw = float(np.linalg.norm(offset_OF_raw))
        if nraw > target_radius and nraw > eps:
            offset_OF_raw *= target_radius / nraw

        if axis_OF_used is not None:
            v1d_scaled = v1d_OF * gain
            offset_OF_proj = axis_OF_used * v1d_scaled
        else:
            offset_OF_proj = np.zeros(2, dtype=np.float32)

        # UI friendly: axis not ready -> proj dot follows raw dot (but arrow still won't show, because no axis)
        if axis_OF_used is None:
            offset_OF_proj = offset_OF_raw.copy()

        # =========================
        # Final output
        # =========================
        if use_mediapipe:
            final_offset = offset_MP_proj
            final_axis   = axis_MP_used
        else:
            if axis_OF_used is None:
                final_offset = offset_OF_raw
                final_axis   = None
            else:
                final_offset = offset_OF_proj
                final_axis   = axis_OF_used

        center_output = center_center + final_offset
        delta = center_output - center_center
        mag = float(np.linalg.norm(delta))
        if mag > target_radius and mag > eps:
            delta *= target_radius / mag
            center_output = center_center + delta

        center_output[0] = np.clip(center_output[0], 0, w - 1)
        center_output[1] = np.clip(center_output[1], 0, h - 1)

        # =========================
        # Draw ROIs
        # =========================
        for (x0, y0, x1, y1) in ROIS:
            cv2.rectangle(gray_bgr, (x0, y0), (x1, y1), (255, 255, 0), 2)

        # =========================
        # Debug text (KEEP AS-IS STYLE)
        # =========================
        y0, dy, line = 15, 14, 0

        cv2.putText(gray_bgr, f"Mode: {mode_str}", (5, y0 + line * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA); line += 1

        cv2.putText(gray_bgr, f"Est. FPS: {sample_fps:.2f}", (5, y0 + line * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1, cv2.LINE_AA); line += 1

        cv2.putText(gray_bgr,
                    f"CheckA(MP): {checkA_mp_trustworthy} vis={mp_avg_vis:.2f} sh={shoulder_width_px:.1f} hip={hip_width_px:.1f} pts={mp_valid_points}",
                    (5, y0 + line * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1, cv2.LINE_AA); line += 1

        cv2.putText(gray_bgr,
                    f"CheckB(CornerCam): {is_camera_moving} mag={corner_mag:.2f} pts={corner_n}",
                    (5, y0 + line * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 255), 1, cv2.LINE_AA); line += 1

        cv2.putText(gray_bgr,
                    f"Hys: ok={mp_ok_now} good={mp_good_cnt} bad={mp_bad_cnt} (N={SWITCH_TO_MP_N}, M={SWITCH_TO_OF_M})",
                    (5, y0 + line * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 200), 1, cv2.LINE_AA); line += 1

        # =========================
        # Draw UI like your sketch (updated):
        #   - project panel has 2 balls:
        #       top  = MP axis
        #       bot  = POA/PCA axis
        #   - blue arrows connect (selected model) -> (selected project axis) -> final
        #     can be diagonal (e.g., model=OF, axis=MP)
        # =========================

        # ---- layout ----
        panel_gap = 28
        panel_w = int(2 * target_radius + 24)

        model_x  = w - debug_margin - target_radius
        proj_x   = model_x - (2 * target_radius + panel_gap)
        final_x  = proj_x  - (2 * target_radius + panel_gap + 40)

        mp_center_model  = np.array([model_x, mp_row_y], dtype=np.float32)
        of_center_model  = np.array([model_x, of_row_y], dtype=np.float32)

        # project panel: top=MP axis, bot=POA axis
        proj_mpaxis_center = np.array([proj_x, mp_row_y], dtype=np.float32)
        proj_poa_center    = np.array([proj_x, of_row_y], dtype=np.float32)

        final_center = np.array([final_x, (mp_row_y + of_row_y) * 0.5], dtype=np.float32)

        # ---- panel boxes ----
        def draw_panel_box(x_center, title):
            x0 = int(x_center - panel_w * 0.5)
            y0 = int(mp_row_y - target_radius - 12)
            x1 = int(x_center + panel_w * 0.5)
            y1 = int(of_row_y + target_radius + 12)
            cv2.rectangle(gray_bgr, (x0, y0), (x1, y1), (120, 120, 120), 2, cv2.LINE_AA)
            cv2.putText(gray_bgr, title, (x0 + 6, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

        draw_panel_box(proj_x,  "project")
        draw_panel_box(model_x, "model")

        # ---- draw circles (model panel + project panel) ----
        for c in [mp_center_model, of_center_model, proj_mpaxis_center, proj_poa_center]:
            cv2.circle(gray_bgr, (int(c[0]), int(c[1])),
                       int(target_radius), (255, 255, 255), 1, cv2.LINE_AA)

        # ---- model dots (so the two model balls are not empty) ----
        # show a tiny status dot: MP ok? / OF always ok
        mp_model_dot = mp_center_model + (offset_MP_proj * 0.35)
        of_model_dot = of_center_model + (offset_OF_raw  * 0.35)  # OF always has raw motion
        mp_model_dot = np.clip(mp_model_dot, [0, 0], [w - 1, h - 1])
        of_model_dot = np.clip(of_model_dot, [0, 0], [w - 1, h - 1])
        cv2.circle(gray_bgr, (int(mp_model_dot[0]), int(mp_model_dot[1])), 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(gray_bgr, (int(of_model_dot[0]), int(of_model_dot[1])), 3, (0, 255, 255), -1, cv2.LINE_AA)

        # ---- project axis arrows (red) ----
        # top project ball: MP axis (if exists)
        draw_axis_arrow(gray_bgr, proj_mpaxis_center, axis_from_mp, target_radius, (0, 0, 255))
        # bottom project ball: POA/PCA axis (if exists)
        draw_axis_arrow(gray_bgr, proj_poa_center, axis_VOF_last, target_radius, (0, 0, 255))

        # ---- helper: projected offset with raw fallback ----
        def proj_offset_with_fallback(V, axis, gain, raw_offset):
            if axis is None:
                return raw_offset.copy()
            v1d = float(np.dot(V, axis))
            return axis * (v1d * gain)

        # =========================
        # NEW UI LOGIC: Source -> Project Panel -> Final copies Project
        # =========================

        # 1) choose CURRENT SOURCE for UI projection
        #    MP mode -> MP raw (V_MP + gain_mp)
        #    OF mode -> OF raw (V_OF + gain)
        #    fallback: if MP not available, use OF
        if use_mediapipe and (fusion_center_mp_ui is not None) and (axis_MP_used is not None):
            V_src        = V_MP.copy()
            gain_src     = float(gain_mp)
            raw_fallback = offset_MP_raw.copy()
        else:
            V_src        = V_OF.copy()
            gain_src     = float(gain)
            raw_fallback = offset_OF_raw.copy()

        def proj_offset_with_fallback_src(V, axis, gain, raw_offset):
            if axis is None:
                return raw_offset.copy()
            v1d = float(np.dot(V, axis))
            return axis * (v1d * gain)

        # 2) Project Panel dots: BOTH balls project the SAME source
        offset_proj_mpaxis = proj_offset_with_fallback_src(V_src, axis_from_mp,  gain_src, raw_fallback)
        offset_proj_poa    = proj_offset_with_fallback_src(V_src, axis_VOF_last, gain_src, raw_fallback)

        mpaxis_dot = np.clip(proj_mpaxis_center + offset_proj_mpaxis, [0, 0], [w - 1, h - 1])
        poa_dot    = np.clip(proj_poa_center    + offset_proj_poa,    [0, 0], [w - 1, h - 1])

        cv2.circle(gray_bgr, (int(mpaxis_dot[0]), int(mpaxis_dot[1])),
                4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(gray_bgr, (int(poa_dot[0]), int(poa_dot[1])),
                4, (0, 255, 255), -1, cv2.LINE_AA)

        # 3) Selection wiring (same decision as before, but Final copies selected Project dot)
        if use_mediapipe:
            sel_model = mp_center_model
            sel_proj  = proj_mpaxis_center
            sel_axis  = axis_from_mp
            sel_project_dot = mpaxis_dot.copy()
        else:
            sel_model = of_center_model
            using_mp_axis_for_of = (checkA_mp_trustworthy and (axis_from_mp is not None))
            if using_mp_axis_for_of:
                sel_proj = proj_mpaxis_center
                sel_axis = axis_from_mp
                sel_project_dot = mpaxis_dot.copy()
            else:
                sel_proj = proj_poa_center
                sel_axis = axis_VOF_last
                sel_project_dot = poa_dot.copy()

        # ---- final big circle ----
        cv2.circle(gray_bgr, (int(final_center[0]), int(final_center[1])),
                int(target_radius), (255, 255, 255), 1, cv2.LINE_AA)

        # final axis arrow (purple)
        draw_axis_arrow(gray_bgr, final_center, sel_axis, target_radius, (255, 0, 255))

        # 4) Final dot = EXACT copy of selected Project dot (copy its offset)
        sel_offset = (sel_project_dot - sel_proj).astype(np.float32)
        final_dot  = np.clip(final_center + sel_offset, [0, 0], [w - 1, h - 1])

        cv2.circle(gray_bgr, (int(final_dot[0]), int(final_dot[1])),
                5, (0, 255, 255), -1, cv2.LINE_AA)

        # ---- dynamic blue arrows (selection wiring) ----
        draw_link_arrow(gray_bgr, sel_model, sel_proj, target_radius, color=(255, 0, 0), thickness=2)
        draw_link_arrow(gray_bgr, sel_proj,  final_center, target_radius, color=(255, 0, 0), thickness=2)

        # ---- highlight selected model circle + selected project circle ----
        def highlight_circle(center):
            cv2.circle(gray_bgr, (int(center[0]), int(center[1])),
                    int(target_radius) + 6, (255, 0, 0), 2, cv2.LINE_AA)

        highlight_circle(sel_model)
        highlight_circle(sel_proj)


        # ---- project dots (yellow) ----
        mpaxis_dot = np.clip(proj_mpaxis_center + offset_proj_mpaxis, [0, 0], [w - 1, h - 1])
        poa_dot    = np.clip(proj_poa_center    + offset_proj_poa,    [0, 0], [w - 1, h - 1])

        cv2.circle(gray_bgr, (int(mpaxis_dot[0]), int(mpaxis_dot[1])),
                   4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(gray_bgr, (int(poa_dot[0]), int(poa_dot[1])),
                   4, (0, 255, 255), -1, cv2.LINE_AA)

        # ---- determine selection (for blue wiring) ----
        # Mode decision:
        #   use_mediapipe=True  -> model=MP, axis=MP
        #   use_mediapipe=False -> model=OF, axis = MP if (MP trustworthy AND MP axis exists), else POA
        if use_mediapipe:
            sel_model = mp_center_model
            sel_proj  = proj_mpaxis_center
            sel_axis  = axis_from_mp
            sel_offset_for_final = offset_MP_proj   # already MP-projected output
        else:
            sel_model = of_center_model
            using_mp_axis_for_of = (checkA_mp_trustworthy and (axis_from_mp is not None))
            if using_mp_axis_for_of:
                sel_proj = proj_mpaxis_center
                sel_axis = axis_from_mp
                sel_offset_for_final = offset_proj_mpaxis
            else:
                sel_proj = proj_poa_center
                sel_axis = axis_VOF_last
                # if POA not ready -> fallback raw for final
                sel_offset_for_final = offset_proj_poa if axis_VOF_last is not None else offset_OF_raw

        # ---- final big circle ----
        cv2.circle(gray_bgr, (int(final_center[0]), int(final_center[1])),
                   int(target_radius), (255, 255, 255), 1, cv2.LINE_AA)

        # final axis arrow (purple)
        draw_axis_arrow(gray_bgr, final_center, sel_axis if (not use_mediapipe) else final_axis,
                       target_radius, (255, 0, 255))

        # final dot uses actually selected path offset
        final_dot = np.clip(final_center + sel_offset_for_final, [0, 0], [w - 1, h - 1])
        cv2.circle(gray_bgr, (int(final_dot[0]), int(final_dot[1])),
                   5, (0, 255, 255), -1, cv2.LINE_AA)

        # ---- dynamic blue arrows (selection wiring) ----
        # model -> selected project axis (may be diagonal) -> final
        draw_link_arrow(gray_bgr, sel_model, sel_proj, target_radius, color=(255, 0, 0), thickness=2)
        draw_link_arrow(gray_bgr, sel_proj,  final_center, target_radius, color=(255, 0, 0), thickness=2)

        # ---- highlight selected model circle + selected project circle ----
        def highlight_circle(center):
            cv2.circle(gray_bgr, (int(center[0]), int(center[1])),
                       int(target_radius) + 6, (255, 0, 0), 2, cv2.LINE_AA)

        highlight_circle(sel_model)
        highlight_circle(sel_proj)

        # =========================
        # END UI
        # =========================
        with lock:
            processed_frame = gray_bgr.copy()
        
    pose.close()



# =========================
# Run threads + display
# =========================
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
            cv2.imshow(
                f"Pose @ {int(TARGET_FPS)} FPS (MP + OF(corner) always-on; OF_full always-on)",
                frame_to_show
            )

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break

except KeyboardInterrupt:
    running = False

t1.join(timeout=1)
t2.join(timeout=1)
cv2.destroyAllWindows()
