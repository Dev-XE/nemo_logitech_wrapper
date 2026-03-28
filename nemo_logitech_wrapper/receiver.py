#!/usr/bin/env python3
import threading
import queue
import time

import rclpy
from rclpy.node import Node

import cv2
import cv2.aruco as aruco
import numpy as np


# ─────────────────────────────────────────────
#  PALETTE & TAG DEFINITIONS
# ─────────────────────────────────────────────
C_ACCENT    = (180, 160, 100)
C_TEXT      = (200, 200, 200)
C_DIM       = (120, 120, 120)
C_OK        = ( 80, 180,  80)
C_WARN      = ( 60, 130, 200)
C_BAD       = ( 70,  70, 180)
C_CROSS     = (160, 160, 160)
C_PIPE      = (200, 200, 160)
C_PANEL_BG  = (18,  18,  18)
C_TITLE     = (160, 140,  90)

PIPELINE_TAGS = {
    56: {"order": 1, "name": "PIPELINE ENTRY", "msg": "Pipeline detected — move to next section"},
    5:  {"order": 2, "name": "SECTION 2",      "msg": "Section 2 reached — proceed forward"},
    20: {"order": 3, "name": "SECTION 3",      "msg": "Section 3 reached — scan surroundings"},
    32: {"order": 4, "name": "PIPELINE EXIT",  "msg": "Pipeline exit — inspection complete"},
}

def get_tag_role(aruco_id):
    if aruco_id in PIPELINE_TAGS:
        return "PIPELINE", PIPELINE_TAGS[aruco_id]
    return "DOCKING", None

# ─────────────────────────────────────────────
#  ARUCO SETUP
# ─────────────────────────────────────────────
ARUCO_DICTS         = [aruco.DICT_4X4_1000, aruco.DICT_5X5_1000,
                       aruco.DICT_6X6_1000, aruco.DICT_7X7_1000]
ARUCO_DICT_NAMES    = ["4x4", "5x5", "6x6", "7x7"]
ARUCO_DICT_PRIORITY = [1, 2, 3, 4]

aruco_params = aruco.DetectorParameters_create()
aruco_params.minMarkerPerimeterRate      = 0.03
aruco_params.maxMarkerPerimeterRate      = 10.0
aruco_params.adaptiveThreshWinSizeMin    = 3
aruco_params.adaptiveThreshWinSizeMax    = 53
aruco_params.adaptiveThreshWinSizeStep   = 4
aruco_params.adaptiveThreshConstant      = 7
aruco_params.minCornerDistanceRate       = 0.05
aruco_params.polygonalApproxAccuracyRate = 0.03
aruco_params.errorCorrectionRate         = 0.6
aruco_params.minMarkerDistanceRate       = 0.05
aruco_params.cornerRefinementMethod      = aruco.CORNER_REFINE_SUBPIX

aruco_dicts = [aruco.getPredefinedDictionary(d) for d in ARUCO_DICTS]

# ─────────────────────────────────────────────
#  SHARED ENHANCEMENT & CONSTANTS
# ─────────────────────────────────────────────
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gamma_lut = np.array([((i / 255.0) ** (1.0 / 1.8)) * 255 for i in range(256)], dtype=np.uint8)

CENTER_TOLERANCE      = 60
MAX_LOST              = 8
NIGHT_THRESHOLD       = 80
MIN_MARKER_AREA       = 500
MARKER_RESULT_TTL     = 0.10
SPATIAL_MERGE_DIST    = 80
DETECT_SCALE          = 0.5
PIPELINE_MSG_COOLDOWN = 3.0
OVERLAY_MSG_DURATION  = 3.0

PIPE_HSV_RANGES = [
    (np.array([18, 90, 70],  dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8)),
    (np.array([ 5, 100, 60], dtype=np.uint8), np.array([18, 255, 255], dtype=np.uint8)),
]
PIPE_MIN_AREA        = 800
PIPE_MIN_ASPECT      = 2.2
PIPE_MIN_FILL_RATIO  = 0.45
PIPE_MIN_SOLIDITY    = 0.65
PIPE_MIN_LENGTH      = 80

# ─────────────────────────────────────────────
#  STATE CLASSES
# ─────────────────────────────────────────────
class ToastMessage:
    def __init__(self):
        self.text, self.sub, self.ts, self.active = "", "", 0.0, False
    def show(self, text, sub=""):
        self.text, self.sub, self.ts, self.active = text, sub, time.monotonic(), True
    def tick(self, now):
        if self.active and (now - self.ts) > OVERLAY_MSG_DURATION:
            self.active = False

class MissionState:
    def __init__(self):
        self.pipeline_found, self.pipeline_angle = False, 0.0
        self.marker_sequence, self.marker_count = [], 0
        self.pinger_side, self._seen_ids_set = "unknown", set()
    def register_tag(self, aruco_id, role):
        if aruco_id not in self._seen_ids_set:
            self._seen_ids_set.add(aruco_id)
            self.marker_sequence.append(aruco_id)
            self.marker_count = len(self.marker_sequence)
            if role == "PIPELINE": self.pipeline_found = True
    def update_pipeline_angle(self, angle_deg): self.pipeline_angle = angle_deg
    def update_pinger_side(self, side): self.pinger_side = side

toast = ToastMessage()
mission = MissionState()

last_marker_result, last_target = None, None
last_marker_ts, lost_frames = 0.0, 0
pipeline_last_msg_ts = {}
prev_time = time.time()

# ─────────────────────────────────────────────
#  HELPERS & LOGIC
# ─────────────────────────────────────────────
def force_put(q, item):
    while True:
        try: q.get_nowait()
        except queue.Empty: break
    try: q.put_nowait(item)
    except queue.Full: pass

def is_valid_marker(pts, scale=1.0):
    area = cv2.contourArea(pts.astype(np.float32))
    if area < MIN_MARKER_AREA * (scale ** 2): return False
    sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    mean = np.mean(sides)
    if mean == 0 or np.any(np.abs(sides - mean) / mean > 0.40): return False
    rect = cv2.minAreaRect(pts.astype(np.float32))
    rw, rh = rect[1]
    if min(rw, rh) == 0 or max(rw, rh) / min(rw, rh) > 1.6: return False
    return True

def handle_pipeline_event(aruco_id, now):
    info = PIPELINE_TAGS[aruco_id]
    if (now - pipeline_last_msg_ts.get(aruco_id, 0.0)) >= PIPELINE_MSG_COOLDOWN:
        pipeline_last_msg_ts[aruco_id] = now
        print(f"\n{'='*55}\n  [PIPELINE][Step {info['order']}/4]  TAG ID: {aruco_id}  |  {info['name']}\n  >> {info['msg']}\n{'='*55}\n")
        toast.show(f"[{info['order']}/4] {info['name']}", info['msg'])
    return info

def handle_docking_event(aruco_id, tx, ty, now):
    print(f"[DOCKING] Tag ID: {aruco_id} | X:{tx} Y:{ty} - Docking marker acquired")
    toast.show(f"DOCKING TAG {aruco_id}", f"X:{tx}  Y:{ty}")

def contour_score_for_pipe(c):
    area = cv2.contourArea(c)
    if area < PIPE_MIN_AREA: return -1, None
    rect = cv2.minAreaRect(c)
    rw, rh = rect[1]
    if rw <= 1 or rh <= 1: return -1, None
    long_side, short_side = max(rw, rh), min(rw, rh)
    if short_side <= 1 or (long_side / short_side) < PIPE_MIN_ASPECT or long_side < PIPE_MIN_LENGTH: return -1, None
    if (area / (rw * rh)) < PIPE_MIN_FILL_RATIO: return -1, None
    hull_area = cv2.contourArea(cv2.convexHull(c))
    if hull_area <= 1 or (area / hull_area) < PIPE_MIN_SOLIDITY: return -1, None
    return (area * 0.002 + (long_side / short_side) * 120 + (area / (rw * rh)) * 200 + (area / hull_area) * 150 + long_side * 0.8), rect

def compute_pipeline_mask(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in PIPE_HSV_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt, best_rect, best_score = None, None, -1
    for c in cnts:
        score, rect = contour_score_for_pipe(c)
        if score > best_score:
            best_score, best_cnt, best_rect = score, c, rect

    mask_vis = np.zeros_like(frame)
    pipe_found, angle, cx_pipe, cy_pipe = False, 0.0, 0, 0

    if best_cnt is not None:
        pipe_found = True
        cv2.drawContours(mask_vis, [best_cnt], -1, (255, 255, 255), -1)
        rw, rh = best_rect[1]
        angle = best_rect[2] + (90 if rw > rh else 0)
        M = cv2.moments(best_cnt)
        if M["m00"] > 0:
            cx_pipe, cy_pipe = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    return mask_vis, pipe_found, angle, cx_pipe, cy_pipe

# ─────────────────────────────────────────────
#  ARUCO THREAD WORKERS
# ─────────────────────────────────────────────
aruco_input_queues = [queue.Queue(maxsize=2) for _ in ARUCO_DICTS]
aruco_result_queue = queue.Queue(maxsize=8)

def make_aruco_worker(det_idx):
    aruco_dict, priority, dict_name, in_q = aruco_dicts[det_idx], ARUCO_DICT_PRIORITY[det_idx], ARUCO_DICT_NAMES[det_idx], aruco_input_queues[det_idx]
    def worker():
        while True:
            try: gray_s, enhanced_s, scale = in_q.get(timeout=1.0)
            except queue.Empty: continue
            corners, ids, _ = aruco.detectMarkers(enhanced_s, aruco_dict, parameters=aruco_params)
            if ids is None: corners, ids, _ = aruco.detectMarkers(gray_s, aruco_dict, parameters=aruco_params)
            if ids is None or len(ids) == 0: continue
            ts = time.monotonic()
            for i in range(len(ids)):
                c = corners[i][0]
                if not is_valid_marker(c, scale): continue
                tx, ty = int(c[:, 0].mean() / scale), int(c[:, 1].mean() / scale)
                aruco_id = int(ids[i][0])
                role, _ = get_tag_role(aruco_id)
                try:
                    aruco_result_queue.put_nowait(({
                        "type": "ARUCO", "priority": priority, "area": cv2.contourArea(c),
                        "tx": tx, "ty": ty, "label": f"ArUco {dict_name}:{aruco_id}",
                        "source": f"ARUCO-{dict_name}", "aruco_id": aruco_id, "role": role,
                        "corner": (corners[i] / scale).astype(np.float32), "aid": ids[i].copy(),
                    }, ts))
                except queue.Full: pass
    return worker

# ─────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────
def draw_panel(frame, x1, y1, x2, y2, color=C_PANEL_BG, alpha=0.65):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_row(frame, label, value, y, val_color=C_TEXT):
    cv2.putText(frame, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, str(value), (160, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, val_color, 1, cv2.LINE_AA)

def draw_crosshair(frame, cx, cy):
    cv2.circle(frame,  (cx, cy), 38, C_CROSS, 1, cv2.LINE_AA)
    cv2.line(frame,   (cx - 22, cy), (cx + 22, cy), C_CROSS, 1)
    cv2.line(frame,   (cx, cy - 22), (cx, cy + 22), C_CROSS, 1)
    cv2.circle(frame,  (cx, cy),  3, C_CROSS, -1)

def draw_toast(frame, now, w, h):
    if not toast.active: return
    toast.tick(now)
    if not toast.active: return
    age = now - toast.ts
    alpha_f = max(0.0, min(1.0, 1.0 - max(0.0, (age - (OVERLAY_MSG_DURATION - 0.6)) / 0.6)))
    card_w, card_h = 500, 72
    x1, y1 = (w - card_w) // 2, h // 2 - card_h // 2 - 30
    x2, y2 = x1 + card_w, y1 + card_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (25, 25, 25), -1)
    cv2.addWeighted(overlay, alpha_f * 0.80, frame, 1 - alpha_f * 0.80, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame, toast.text, (x1 + 14, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_ACCENT, 1, cv2.LINE_AA)
    if toast.sub: cv2.putText(frame, toast.sub, (x1 + 14, y1 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_TEXT, 1, cv2.LINE_AA)

# ─────────────────────────────────────────────
#  ROS 2 NODE CONFIGURATION
# ─────────────────────────────────────────────
class GstRxNode(Node):
    def __init__(self):
        super().__init__("gst_rx_node")
        self.declare_parameter("port", 5001)
        self.declare_parameter("latency_ms", 10)
        self.declare_parameter("sync", False)
        
    def get_pipeline_string(self) -> str:
        # EXACT same pipeline parameters, but routing to OpenCV (appsink) instead of screen (autovideosink)
        port = int(self.get_parameter("port").value)
        latency = int(self.get_parameter("latency_ms").value)
        sync = bool(self.get_parameter("sync").value)
        sync_str = "true" if sync else "false"
        
        pipeline = (
            f'udpsrc port={port} caps="application/x-rtp, media=video, encoding-name=H264, payload=96" ! '
            f'rtpjitterbuffer latency={latency} ! '
            f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! '
            f'video/x-raw, format=BGR ! '
            f'appsink drop=True max-buffers=1 sync={sync_str}'
        )
        self.get_logger().info(f"OpenCV pulling stream via GStreamer:\n{pipeline}")
        return pipeline

# ─────────────────────────────────────────────
#  DETECTION MAIN LOOP
# ─────────────────────────────────────────────
def detection_main(cap):
    global last_marker_result, last_marker_ts, last_target, lost_frames, prev_time
    for i in range(len(ARUCO_DICTS)):
        threading.Thread(target=make_aruco_worker(i), daemon=True).start()

    print("[INFO] Professional Pipeline + ArUco Detection UI (Receiving from UDP Stream)")
    print("[INFO] Press Q to quit\n")

    prev_detected_id = None
    while True:
        ret, frame = cap.read()
        if not ret:
            # UDP stream might just be waiting for the transmitter.
            time.sleep(0.01)
            continue

        h, w   = frame.shape[:2]
        cx, cy = w // 2, h // 2
        now    = time.monotonic()
        current_time = time.time()
        fps      = 1 / max((current_time - prev_time), 1e-6)
        prev_time = current_time

        mask_vis, pipe_found, pipe_angle, pipe_cx, pipe_cy = compute_pipeline_mask(frame)
        if pipe_found:
            mission.update_pipeline_angle(pipe_angle)
            mission.pipeline_found = True
            mission.update_pinger_side("right" if pipe_cx > cx else "left")

        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = cv2.mean(gray)[0]
        if brightness < NIGHT_THRESHOLD:
            enhanced = clahe.apply(cv2.LUT(gray, gamma_lut))
            mode = "NIGHT"
        else:
            enhanced = cv2.equalizeHist(gray)
            mode = "DAY"

        gray_s     = cv2.resize(gray,     (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
        enhanced_s = cv2.resize(enhanced, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)

        for q in aruco_input_queues:
            force_put(q, (gray_s, enhanced_s, DETECT_SCALE))

        fresh_aruco = []
        while True:
            try:
                r, ts = aruco_result_queue.get_nowait()
                if (now - ts) < MARKER_RESULT_TTL: fresh_aruco.append((r, ts))
            except queue.Empty: break

        if fresh_aruco:
            all_det = [(r["priority"], r["area"], r["tx"], r["ty"], r) for r, _ in fresh_aruco]
            used, groups = [False] * len(all_det), []
            for i in range(len(all_det)):
                if used[i]: continue
                group = [i]
                used[i] = True
                for j in range(i + 1, len(all_det)):
                    if not used[j] and ((all_det[i][2]-all_det[j][2])**2 + (all_det[i][3]-all_det[j][3])**2)**0.5 < SPATIAL_MERGE_DIST:
                        group.append(j); used[j] = True
                groups.append(group)

            best_r, best_score = None, (-1, -1)
            for group in groups:
                for idx in group:
                    if (all_det[idx][0], all_det[idx][1]) > best_score:
                        best_score, best_r = (all_det[idx][0], all_det[idx][1]), all_det[idx][4]

            if best_r:
                last_marker_result, last_marker_ts = best_r, now
                aid, role = best_r["aruco_id"], best_r["role"]
                mission.register_tag(aid, role)
                if role == "PIPELINE": handle_pipeline_event(aid, now)
                elif role == "DOCKING" and aid != prev_detected_id: handle_docking_event(aid, best_r["tx"], best_r["ty"], now)
                prev_detected_id = aid

        detected = None
        if last_marker_result is not None and (now - last_marker_ts) < MARKER_RESULT_TTL:
            r = last_marker_result
            detected = (r["tx"], r["ty"], r["label"], r["role"], r["aruco_id"])
            aruco.drawDetectedMarkers(frame, [r["corner"]], np.array([[r["aruco_id"]]]))
        elif (now - last_marker_ts) >= MARKER_RESULT_TTL: last_marker_result = None

        if detected is not None:
            last_target, lost_frames = detected, 0
        else:
            lost_frames += 1
            if lost_frames >= MAX_LOST: last_target, last_marker_result, prev_detected_id = None, None, None

        # ── HUD LAYOUT ──
        PANEL_LEFT_W, PANEL_LEFT_Y1, PANEL_LEFT_Y2 = 260, 58, 286
        draw_panel(frame, 0, 0, w, 48)
        draw_panel(frame, 8, PANEL_LEFT_Y1, PANEL_LEFT_W, PANEL_LEFT_Y2)
        draw_panel(frame, 0, h - 44, w, h)

        cv2.putText(frame, "AUTONOMOUS PIPELINE INSPECTION", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, C_TITLE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"MODE: {mode}", (w - 190, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_OK if mode == "DAY" else C_WARN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"BRIGHT: {int(brightness)}", (w - 190, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_DIM,  1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 80,  20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_OK,   1, cv2.LINE_AA)

        cv2.putText(frame, "TELEMETRY", (18, PANEL_LEFT_Y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_TITLE, 1, cv2.LINE_AA)
        cv2.line(frame, (18, PANEL_LEFT_Y1 + 30), (PANEL_LEFT_W - 14, PANEL_LEFT_Y1 + 30), C_TITLE, 1)

        ROW_START, ROW_GAP = PANEL_LEFT_Y1 + 52, 24
        if last_target is not None and lost_frames < MAX_LOST:
            tx, ty, label, role, aid = last_target
            dx, dy = tx - cx, ty - cy
            centered = abs(dx) < CENTER_TOLERANCE and abs(dy) < CENTER_TOLERANCE

            draw_row(frame, "TARGET ID", aid, ROW_START, C_ACCENT)
            draw_row(frame, "ROLE", role, ROW_START + ROW_GAP, C_WARN)
            draw_row(frame, "DX", dx, ROW_START + ROW_GAP*2, C_TEXT)
            draw_row(frame, "DY", dy, ROW_START + ROW_GAP*3, C_TEXT)
            draw_row(frame, "PINGER", mission.pinger_side.upper(), ROW_START + ROW_GAP*4, C_TEXT)
            draw_row(frame, "PIPE ANG", f"{mission.pipeline_angle:.1f}d", ROW_START + ROW_GAP*5, C_TEXT)
            cv2.putText(frame, "CENTER LOCKED" if centered else "ALIGNING", (18, ROW_START + ROW_GAP*6 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_OK if centered else C_BAD, 1, cv2.LINE_AA)

            cv2.circle(frame, (tx, ty), 6, C_OK, -1)
            cv2.rectangle(frame, (tx - 22, ty - 22), (tx + 22, ty + 22), C_OK, 1)
            cv2.arrowedLine(frame, (cx, cy), (tx, ty), C_DIM, 1, tipLength=0.12)
            cv2.putText(frame, f"ID {aid}", (tx + 14, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_ACCENT, 1, cv2.LINE_AA)
        else:
            for i, (l, v) in enumerate([("TARGET ID", "--"), ("ROLE", "NONE"), ("DX", "--"), ("DY", "--"), ("PINGER", mission.pinger_side.upper()), ("PIPE ANG", f"{mission.pipeline_angle:.1f}d")]):
                draw_row(frame, l, v, ROW_START + ROW_GAP*i, C_BAD if i < 2 else C_DIM)
            cv2.putText(frame, "NO TARGET", (18, ROW_START + ROW_GAP*6 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_BAD, 1, cv2.LINE_AA)

        draw_crosshair(frame, cx, cy)
        if pipe_found:
            cv2.circle(frame, (pipe_cx, pipe_cy), 5, C_PIPE, -1)
            cv2.putText(frame, "PIPE", (pipe_cx + 8, pipe_cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_PIPE, 1, cv2.LINE_AA)

        cv2.putText(frame, f"STATUS: {'PIPELINE DETECTED' if mission.pipeline_found else 'SEARCHING PIPELINE'}", (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_OK if mission.pipeline_found else C_BAD, 1, cv2.LINE_AA)
        cv2.putText(frame, f"MARKERS: {mission.marker_count}", (w - 160, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_DIM, 1, cv2.LINE_AA)

        draw_toast(frame, now, w, h)

        cv2.imshow("Pipeline Inspection Feed", frame)
        cv2.imshow("Pipeline Mask", mask_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ─────────────────────────────────────────────
#  MAIN EXECUTION
# ─────────────────────────────────────────────
def main():
    rclpy.init()
    node = GstRxNode()

    # Spin ROS 2 network tasks in a background thread so OpenCV owns the main thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Start the OpenCV capture using the parameters pulled securely from ROS
    pipeline_str = node.get_pipeline_string()
    cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        node.get_logger().error("Failed to open UDP stream via GStreamer! Make sure your OpenCV installation is built with GStreamer support.")
        return

    try:
        detection_main(cap)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()  
