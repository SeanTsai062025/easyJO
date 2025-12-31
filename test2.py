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
VIDEO_PATH = "test_video/side.MOV"
TARGET_FPS = 10.0

SMA_WINDOW = 3          # Step 2 baseline smoothing window (V=P-B to find the B)
ENV_WINDOW = 3          # Step 5 envelope window (to find the amplitude)

# ---- Check A: MP trustworthy ----
MP_VIS_THRESH = 0.7     # avg visibility threshold

# 點不要擠在一起：用肩寬 / 髖寬判斷（像素）
MIN_SHOULDER_WIDTH_PX = 35
MIN_HIP_WIDTH_PX      = 30

MP_CHECKA_MIN_POINTS = 12  # from the 14 selected points, need >=12 to run CheckA

# ---- Check B: Corner-based camera motion detection (只負責判斷鏡頭動不動) ----
CORNER_RATIO      = 0.15
CORNER_MOVE_PX    = 0.9
CORNER_MIN_POINTS = 12

# ---- Mode hysteresis (anti flicker) ----
SWITCH_TO_MP_N = 1   # FLOW -> MP needs N consecutive mp_ok_now=True
SWITCH_TO_OF_M = 5   # MP -> FLOW needs M consecutive mp_ok_now=False

# ---- MP UI always show ----
MP_UI_MIN_POINTS = 4
TARGET_RADIUS = 30.0

# ---- tracking mediapipe landmark ----
VISIBLE_POINTS = list(range(1, 13)) + [23, 24]   # 14 points
# =========================
# tunable parameters end
# =========================


# =========================
# Global shared buffers
# =========================
latest_frame = None
processed_frame = None
latest_frame_id = -1
lock = threading.Lock()
running = True


def draw_selected_landmarks(image, landmarks):
    """Draw only selected landmark points (no skeleton lines)."""
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

    if src_fps > TARGET_FPS:
        frames_to_skip = int(src_fps / TARGET_FPS) - 1
        frames_to_skip = max(0, frames_to_skip)
    else:
        frames_to_skip = 0

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
    cut_thresh = 0.1 * min(SMALL_W, SMALL_H)

    # MP pipeline buffers
    window_P_MP = deque(maxlen=SMA_WINDOW)
    window_M_MP = deque(maxlen=ENV_WINDOW)
    envelope_MP = 0.0
    prev_P_MP = None

    # OF output pipeline buffers (full-frame OF output)
    window_P_OF = deque(maxlen=SMA_WINDOW)
    window_M_OF = deque(maxlen=ENV_WINDOW)
    envelope_OF = 0.0
    prev_P_virtual = None

    # ===== Optical Flow states =====
    # (1) OF_cam: corner ROI for camera motion only
    prev_gray_cam = None
    prev_pts_cam  = None

    # (2) OF_full: full-frame flow for output
    prev_gray_full = None
    prev_pts_full  = None
    P_virtual      = None

    # Axis memory (from MP) -> used as primary axis when available
    axis_MP_last = None

    # MP UI: hold last offsets so dots always show
    mp_ui_offset_proj_last = np.zeros(2, dtype=np.float32)
    mp_ui_offset_raw_last  = np.zeros(2, dtype=np.float32)

    # ---- Mode hysteresis states ----
    mode_use_mp = False
    mp_good_cnt = 0
    mp_bad_cnt  = 0

    # LK optical flow params
    FLOW_MAX_CORNERS = 80
    FLOW_QUALITY     = 0.01
    FLOW_MIN_DIST    = 7

    # Debug UI layout
    debug_margin = 10
    target_radius = float(TARGET_RADIUS)
    row_spacing = 2 * target_radius + 20

    mp_row_y = debug_margin + target_radius
    of_row_y = mp_row_y + row_spacing

    # ===== New: axis from V_OF(t) (baseline-removed motion) =====
    AXIS_WIN = 20          # number of recent V_OF samples (10fps => ~2s)
    AXIS_MIN_SAMPLES = 8   # minimum samples to estimate axis
    AXIS_MIN_ENERGY = 1.0  # if motion energy too small, do not update axis
    AXIS_EMA_ALPHA = 0.25  # smoothing for axis (0.1~0.3)

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
        """
        Estimate principal oscillation axis from baseline-removed motion V(t).
        Returns a unit vector (2,) or None.
        """
        if len(windowV) < AXIS_MIN_SAMPLES:
            return None

        X = np.array(windowV, dtype=np.float32)  # (N,2)

        # motion too small -> do not update
        energy = float(np.mean(np.sum(X * X, axis=1)))
        if energy < AXIS_MIN_ENERGY:
            return None

        Xm = X - np.mean(X, axis=0, keepdims=True)
        C = (Xm.T @ Xm) / max(1, Xm.shape[0])    # (2,2)

        evals, evecs = np.linalg.eigh(C)         # ascending
        axis = evecs[:, int(np.argmax(evals))].astype(np.float32)

        n = float(np.linalg.norm(axis))
        if n < 1e-6:
            return None
        axis /= n

        # anti-flip
        if prev_axis is not None and float(np.dot(axis, prev_axis)) < 0:
            axis = -axis

        return axis

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

        # resize & grayscale (optical flow uses gray only)
        small = cv2.resize(frame_copy, (SMALL_W, SMALL_H))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray_rgb = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2RGB)

        h, w, _ = gray_bgr.shape
        center_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        # ===== Corner ROIs for CheckB (camera motion) =====
        roi_w = int(w * CORNER_RATIO)
        roi_h = int(h * CORNER_RATIO)
        ROIS = [
            (0, 0, roi_w, roi_h),                 # TL
            (w - roi_w, 0, w, roi_h),             # TR
            (0, h - roi_h, roi_w, h),             # BL
            (w - roi_w, h - roi_h, w, h),         # BR
        ]
        corner_mask = np.zeros_like(gray, dtype=np.uint8)
        for (x0, y0, x1, y1) in ROIS:
            corner_mask[y0:y1, x0:x1] = 255

        # UI small circles centers
        mp_proj_center = np.array([w - debug_margin - 3 * target_radius, mp_row_y], dtype=np.float32)
        mp_raw_center  = np.array([w - debug_margin - 1 * target_radius, mp_row_y], dtype=np.float32)
        of_proj_center = np.array([w - debug_margin - 3 * target_radius, of_row_y], dtype=np.float32)
        of_raw_center  = np.array([w - debug_margin - 1 * target_radius, of_row_y], dtype=np.float32)

        # =========================
        # MediaPipe measurements (for UI + CheckA + axis source)
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

            # MP UI centroid: allow partial
            if mp_valid_points >= MP_UI_MIN_POINTS:
                fusion_center_mp_ui = np.array([float(np.mean(xs_in)), float(np.mean(ys_in))], dtype=np.float32)

            # CheckA needs enough points
            if mp_valid_points >= MP_CHECKA_MIN_POINTS and fusion_center_mp_ui is not None:
                mp_avg_vis = float(np.mean(vis_in))
                fusion_center_mp_check = fusion_center_mp_ui

                # shoulder/hip width
                ls = lms[11]; rs = lms[12]
                lh = lms[23]; rh = lms[24]

                ls_xy = np.array([ls.x * SMALL_W, ls.y * SMALL_H], dtype=np.float32)
                rs_xy = np.array([rs.x * SMALL_W, rs.y * SMALL_H], dtype=np.float32)
                lh_xy = np.array([lh.x * SMALL_W, lh.y * SMALL_H], dtype=np.float32)
                rh_xy = np.array([rh.x * SMALL_W, rh.y * SMALL_H], dtype=np.float32)

                shoulder_width_px = float(np.linalg.norm(ls_xy - rs_xy))
                hip_width_px      = float(np.linalg.norm(lh_xy - rh_xy))

            # anatomical axis from MP (nose -> shoulders mid)
            nose = lms[0]
            nx, ny = nose.x * SMALL_W, nose.y * SMALL_H
            if 0 <= nx < SMALL_W and 0 <= ny < SMALL_H:
                nose_pt = np.array([nx, ny], dtype=np.float32)

            lh = lms[11]
            rh = lms[12]
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
        # Check B (OF_cam): Corner-only optical flow -> camera moving?
        # =========================
        is_camera_moving = False
        corner_mag = 0.0
        corner_n = 0

        if prev_gray_cam is None:
            prev_gray_cam = gray.copy()
            prev_pts_cam = cv2.goodFeaturesToTrack(
                prev_gray_cam,
                maxCorners=FLOW_MAX_CORNERS,
                qualityLevel=FLOW_QUALITY,
                minDistance=FLOW_MIN_DIST,
                blockSize=7,
                mask=corner_mask,
            )
        else:
            if prev_pts_cam is not None and len(prev_pts_cam) > 0:
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray_cam,
                    gray,
                    prev_pts_cam,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )

                if next_pts is not None and status is not None:
                    good_new = next_pts[status.reshape(-1) == 1]
                    good_old = prev_pts_cam[status.reshape(-1) == 1]
                    corner_n = len(good_new)

                    if corner_n >= CORNER_MIN_POINTS:
                        diffs = (good_new - good_old).reshape(-1, 2).astype(np.float32)
                        v_corner = np.median(diffs, axis=0).astype(np.float32)
                        corner_mag = float(np.linalg.norm(v_corner))
                        is_camera_moving = (corner_mag > CORNER_MOVE_PX)
                    else:
                        is_camera_moving = False

            prev_gray_cam = gray.copy()
            prev_pts_cam = cv2.goodFeaturesToTrack(
                prev_gray_cam,
                maxCorners=FLOW_MAX_CORNERS,
                qualityLevel=FLOW_QUALITY,
                minDistance=FLOW_MIN_DIST,
                blockSize=7,
                mask=corner_mask,
            )

        # =========================
        # Optical Flow output (OF_full): Full-frame optical flow -> v_flow_full -> P_virtual
        # =========================
        v_flow_full = np.zeros(2, dtype=np.float32)

        if prev_gray_full is None:
            prev_gray_full = gray.copy()
            prev_pts_full = cv2.goodFeaturesToTrack(
                prev_gray_full,
                maxCorners=FLOW_MAX_CORNERS,
                qualityLevel=FLOW_QUALITY,
                minDistance=FLOW_MIN_DIST,
                blockSize=7,
                mask=None,
            )
        else:
            if prev_pts_full is not None and len(prev_pts_full) > 0:
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray_full,
                    gray,
                    prev_pts_full,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )

                if next_pts is not None and status is not None:
                    good_new = next_pts[status.reshape(-1) == 1]
                    good_old = prev_pts_full[status.reshape(-1) == 1]
                    if len(good_new) > 0:
                        diffs_full = (good_new - good_old).reshape(-1, 2).astype(np.float32)
                        v_flow_full = np.median(diffs_full, axis=0).astype(np.float32)

            prev_gray_full = gray.copy()
            prev_pts_full = cv2.goodFeaturesToTrack(
                prev_gray_full,
                maxCorners=FLOW_MAX_CORNERS,
                qualityLevel=FLOW_QUALITY,
                minDistance=FLOW_MIN_DIST,
                blockSize=7,
                mask=None,
            )

        # integrate P_virtual using full-frame flow
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
        # Check A (MP trustworthy): visibility + not-collapsed
        # =========================
        checkA_mp_trustworthy = False
        if fusion_center_mp_check is not None:
            cond_vis = (mp_avg_vis > MP_VIS_THRESH)
            cond_not_collapsed = (shoulder_width_px >= MIN_SHOULDER_WIDTH_PX) and (hip_width_px >= MIN_HIP_WIDTH_PX)
            checkA_mp_trustworthy = (cond_vis and cond_not_collapsed)

        # this frame recommends MP?
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
        # Compute MP UI offsets + OF offsets
        # =========================
        offset_MP_proj = mp_ui_offset_proj_last.copy()
        offset_MP_raw  = mp_ui_offset_raw_last.copy()
        axis_MP_used   = axis_from_mp

        offset_OF_proj = np.zeros(2, dtype=np.float32)
        offset_OF_raw  = np.zeros(2, dtype=np.float32)

        # ===== Axis selection for OF =====
        axis_OF_used = axis_from_mp  # default: use MP axis if available

        # New rule: if using FLOW and MP is NOT trustworthy -> use axis learned from V_OF(t)
        if (not use_mediapipe) and (not checkA_mp_trustworthy):
            axis_OF_used = axis_VOF_last  # may be None -> then we will fallback to raw later

        # ---- MP pipeline (UI) ----
        if fusion_center_mp_ui is not None:
            P_mp = fusion_center_mp_ui.astype(np.float32)

            reset = maybe_reset_on_cut(P_mp, prev_P_MP, window_P_MP, window_M_MP)
            if reset:
                envelope_MP = 0.0
            prev_P_MP = P_mp.copy()

            window_P_MP.append(P_mp.copy())
            baseline_MP = np.mean(np.array(window_P_MP), axis=0).astype(np.float32)
            V_MP = (P_mp - baseline_MP).astype(np.float32)

            v1d_MP = 0.0
            if axis_MP_used is not None:
                v1d_MP = float(np.dot(V_MP, axis_MP_used))

            window_M_MP.append(abs(v1d_MP))
            envelope_MP = max(window_M_MP) if len(window_M_MP) > 0 else 0.0

            if envelope_MP > eps:
                gain = target_radius / envelope_MP
                v1d_scaled = v1d_MP * gain
            else:
                gain = 0.0
                v1d_scaled = 0.0

            if axis_MP_used is not None:
                offset_MP_proj = axis_MP_used * v1d_scaled
            else:
                offset_MP_proj = np.zeros(2, dtype=np.float32)

            offset_MP_raw = V_MP * gain
            nraw = float(np.linalg.norm(offset_MP_raw))
            if nraw > target_radius and nraw > eps:
                offset_MP_raw *= target_radius / nraw

            mp_ui_offset_proj_last = offset_MP_proj.copy()
            mp_ui_offset_raw_last  = offset_MP_raw.copy()

        # ---- OF output pipeline (full-frame P_virtual) ----
        P_of = P_virtual.astype(np.float32)

        if prev_P_virtual is not None:
            dist = float(np.linalg.norm(P_of - prev_P_virtual))
            if dist > cut_thresh:
                window_P_OF.clear()
                window_M_OF.clear()
                envelope_OF = 0.0
                window_V_OF.clear()  # also reset axis window on hard cut
        prev_P_virtual = P_of.copy()

        window_P_OF.append(P_of.copy())
        baseline_OF = np.mean(np.array(window_P_OF), axis=0).astype(np.float32)
        V_OF = (P_of - baseline_OF).astype(np.float32)

        # ===== New: update axis estimator from V_OF(t) (after we computed V_OF) =====
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

        v1d_OF = 0.0
        if axis_OF_used is not None:
            v1d_OF = float(np.dot(V_OF, axis_OF_used))

        window_M_OF.append(abs(v1d_OF))
        envelope_OF = max(window_M_OF) if len(window_M_OF) > 0 else 0.0

        if envelope_OF > eps:
            gain = target_radius / envelope_OF
            v1d_scaled = v1d_OF * gain
        else:
            gain = 0.0
            v1d_scaled = 0.0

        if axis_OF_used is not None:
            offset_OF_proj = axis_OF_used * v1d_scaled
        else:
            offset_OF_proj = np.zeros(2, dtype=np.float32)

        offset_OF_raw = V_OF * gain
        nraw = float(np.linalg.norm(offset_OF_raw))
        if nraw > target_radius and nraw > eps:
            offset_OF_raw *= target_radius / nraw

        # =========================
        # Final output uses selected mode
        # =========================
        if use_mediapipe:
            final_offset = offset_MP_proj
            final_axis   = axis_MP_used
        else:
            # If no reliable axis (MP untrustworthy + axis_VOF_last not ready), fallback to raw
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
        # UI: draw ROIs used for CheckB
        # =========================
        for (x0, y0, x1, y1) in ROIS:
            cv2.rectangle(gray_bgr, (x0, y0), (x1, y1), (255, 255, 0), 2)

        # =========================
        # Debug text
        # =========================
        y0, dy, line = 15, 14, 0

        cv2.putText(gray_bgr, f"Mode: {mode_str}", (5, y0 + line * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
        line += 1

        cv2.putText(gray_bgr, f"Est. FPS: {sample_fps:.2f}", (5, y0 + line * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1, cv2.LINE_AA)
        line += 1

        cv2.putText(gray_bgr,
                    f"CheckA(MP): {checkA_mp_trustworthy}  vis={mp_avg_vis:.2f} sh={shoulder_width_px:.1f} hip={hip_width_px:.1f} pts={mp_valid_points}",
                    (5, y0 + line * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1, cv2.LINE_AA)
        line += 1

        cv2.putText(gray_bgr,
                    f"CheckB(CornerCam): {is_camera_moving}  mag={corner_mag:.2f} pts={corner_n}",
                    (5, y0 + line * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 255), 1, cv2.LINE_AA)
        line += 1

        cv2.putText(gray_bgr,
                    f"Hys: ok={mp_ok_now} good={mp_good_cnt} bad={mp_bad_cnt} (N={SWITCH_TO_MP_N}, M={SWITCH_TO_OF_M})",
                    (5, y0 + line * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 200), 1, cv2.LINE_AA)
        line += 1

        ax_ready = (axis_VOF_last is not None)
        cv2.putText(gray_bgr,
                    f"Axis(V_OF) ready={ax_ready}  MPenv={envelope_MP:.3f} OFenv={envelope_OF:.3f}",
                    (5, y0 + line * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 255), 1, cv2.LINE_AA)

        # =========================
        # Draw circles and dots
        # =========================
        # 1) Big final circle
        cv2.circle(gray_bgr, (int(center_center[0]), int(center_center[1])),
                   int(target_radius), (255, 255, 255), 1, cv2.LINE_AA)
        draw_axis_arrow(gray_bgr, center_center, final_axis, target_radius, (255, 0, 255))
        cv2.circle(gray_bgr, (int(center_output[0]), int(center_output[1])),
                   5, (0, 255, 255), -1, cv2.LINE_AA)

        # 2) MP two small circles
        cv2.circle(gray_bgr, (int(mp_proj_center[0]), int(mp_proj_center[1])),
                   int(target_radius), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(gray_bgr, (int(mp_raw_center[0]), int(mp_raw_center[1])),
                   int(target_radius), (255, 255, 255), 1, cv2.LINE_AA)

        draw_axis_arrow(gray_bgr, mp_proj_center, axis_MP_used, target_radius, (0, 0, 255))
        mp_proj_dot = mp_proj_center + offset_MP_proj
        mp_raw_dot  = mp_raw_center + offset_MP_raw
        mp_proj_dot = np.clip(mp_proj_dot, [0, 0], [w - 1, h - 1])
        mp_raw_dot  = np.clip(mp_raw_dot,  [0, 0], [w - 1, h - 1])
        cv2.circle(gray_bgr, (int(mp_proj_dot[0]), int(mp_proj_dot[1])),
                   4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(gray_bgr, (int(mp_raw_dot[0]), int(mp_raw_dot[1])),
                   4, (0, 255, 255), -1, cv2.LINE_AA)

        # 3) OF two small circles
        cv2.circle(gray_bgr, (int(of_proj_center[0]), int(of_proj_center[1])),
                   int(target_radius), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(gray_bgr, (int(of_raw_center[0]), int(of_raw_center[1])),
                   int(target_radius), (255, 255, 255), 1, cv2.LINE_AA)

        draw_axis_arrow(gray_bgr, of_proj_center, axis_OF_used, target_radius, (0, 0, 255))
        of_proj_dot = of_proj_center + offset_OF_proj
        of_raw_dot  = of_raw_center + offset_OF_raw
        of_proj_dot = np.clip(of_proj_dot, [0, 0], [w - 1, h - 1])
        of_raw_dot  = np.clip(of_raw_dot, [0, 0], [w - 1, h - 1])
        cv2.circle(gray_bgr, (int(of_proj_dot[0]), int(of_proj_dot[1])),
                   4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(gray_bgr, (int(of_raw_dot[0]), int(of_raw_dot[1])),
                   4, (0, 255, 255), -1, cv2.LINE_AA)

        # 4) Highlight selected row
        highlight_color  = (255, 0, 0)
        highlight_thick  = 2
        highlight_margin = 6

        if use_mediapipe:
            row_y   = mp_row_y
            left_x  = mp_proj_center[0]
            right_x = mp_raw_center[0]
        else:
            row_y   = of_row_y
            left_x  = of_proj_center[0]
            right_x = of_raw_center[0]

        x_min = min(left_x, right_x) - target_radius - highlight_margin
        x_max = max(left_x, right_x) + target_radius + highlight_margin
        y_min = row_y - target_radius - highlight_margin
        y_max = row_y + target_radius + highlight_margin

        x_min = int(max(0, x_min))
        x_max = int(min(w - 1, x_max))
        y_min = int(max(0, y_min))
        y_max = int(min(h - 1, y_max))

        cv2.rectangle(gray_bgr, (x_min, y_min), (x_max, y_max),
                      highlight_color, highlight_thick, cv2.LINE_AA)

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
                f"Pose @ {int(TARGET_FPS)} FPS (CheckB: corner-only, OF output: full-frame)",
                frame_to_show
            )

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
