"""
=================================================================================
LAB 2: MAZE TARGET RUNNER (Round 2)
=================================================================================
Description:
- ใช้ผลสำรวจจาก LAB 1 (map_state_run1_*.json + targets_run1_*.json)
- โหลดกราฟ/กำแพง/จุดที่สำรวจ แล้ววิ่งด้วย shortest path ไปยังเป้าหมาย
- เล็งและยิง "ไม่ใช้ ToF ระหว่างเล็ง" (นิ่ง ไม่ขยับตัวรถตอนเล็ง)
- ไม่สแกนเพิ่มในรอบ 2 (เชื่อถือแผนที่จากรอบแรก)

Usage:
1) รัน LAB 1 ให้สำรวจเสร็จ จะได้ไฟล์:
   - map_state_run1_YYYYMMDD_HHMMSS.json
   - targets_run1_YYYYMMDD_HHMMSS.json (ถ้าคุณเรียก record_target_hit)
2) ตั้งชื่อไฟล์ทั้งสองด้านล่าง แล้วรัน LAB 2

Author: [Your Name]
Date: October 2025
=================================================================================
"""

import json
import time
import math
from collections import deque
from datetime import datetime

# ==== ดึงของเดิมจาก LAB 1 ====
from test_lap1 import (
    robot,                      # robomaster.robot
    MazeExplorer,               # คลาสหลักจาก LAB1
    GRID_SIZE_M,                # ขนาดกริด
    plot_map_with_walls,        # ฟังก์ชันวาดแผนที่ (เผื่ออยากบันทึก)
)

# ====== ตั้งชื่อไฟล์ JSON (แก้ให้ตรงกับไฟล์ที่ได้จาก LAB 1) ======
MAP_JSON_PATH = "map_state_run1_YYYYMMDD_HHMMSS.json"
TARGETS_JSON_PATH = "targets_run1_YYYYMMDD_HHMMSS.json"

# ====== Utility: สร้างกราฟ/BFS จาก JSON ======
def _graph_from_json(map_payload):
    # key "graph" : {"x,y": [[nx,ny], ...], ...}
    g = {}
    for k, lst in map_payload["graph"].items():
        x, y = map(int, k.split(","))
        g[(x, y)] = set(tuple(n) for n in lst)
    return g

def _bfs_path(graph, start, goal):
    if start == goal:
        return [start]
    q = deque([(start, [start])])
    visited = {start}
    while q:
        cur, path = q.popleft()
        for nb in graph.get(cur, []):
            if nb in visited:
                continue
            if nb == goal:
                return path + [nb]
            visited.add(nb)
            q.append((nb, path + [nb]))
    return None

# ====== LAB 2 Runner (สืบทอดจาก LAB 1) ======
class MazeTargetRunner(MazeExplorer):
    """
    รอบ 2: ใช้แผนที่สำเร็จรูปจาก LAB 1 เพื่อไปยังเป้าหมายแบบ shortest path
    - ไม่สแกนระหว่างทาง
    - เล็ง/ยิงแบบไม่ใช้ ToF gating
    """

    # --- เล็งและยิงแบบไม่ใช้ ToF ระหว่างเล็ง ---
    def detect_and_shot_no_tof(self, timeout_s=8.0):
        pid_yaw = PID(kp=0.20, ki=0.00, kd=0.00, integ_clip=40.0, out_clip=10.0)
        pid_pitch = PID(kp=0.20, ki=0.00, kd=0.00, integ_clip=40.0, out_clip=8.0)

        self.ep_gimbal.recenter().wait_for_completed()
        self.ep_gimbal.drive_speed(0,0)  # ensure stop
        time.sleep(0.2)

        t0 = time.monotonic()
        have_object = False
        self.ep_camera.start_video_stream(display=False, resolution="720p")

        center_x = center_y = None
        err_x = err_y = 0.0
        self.last_time = time.monotonic()

        while True:
            # === อ่านภาพ ===
            try:
                img = self.ep_camera.read_cv2_image(strategy="newest", timeout=1)
                if img is None:
                    if time.monotonic() - t0 > timeout_s: break
                    continue
            except Exception:
                if time.monotonic() - t0 > timeout_s: break
                continue

            # === เหมือน LAB1: สร้าง mask หา shape ตาม target_color/target_shape ===
            import cv2, numpy as np  # local import
            color_key = self.target_color
            if self.target_color in self.colors:
                color_key = self.colors[self.target_color]
            if color_key not in self.color_ranges:
                print(f"⚠️ ไม่พบสี '{self.target_color}'")
                break

            h, w = img.shape[:2]
            blur = cv2.GaussianBlur(img, (5,5), 0)
            hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            lower = np.array(self.color_ranges[color_key][0])
            upper = np.array(self.color_ranges[color_key][1])
            mask  = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            center_x = center_y = None
            for cnt in contours:
                if cv2.contourArea(cnt) < 200:
                    continue
                per = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * per, True)
                vtx = len(approx)

                shape = None
                if vtx >= 8:
                    area = cv2.contourArea(cnt)
                    circ = 4*math.pi*area/(per*per + 1e-6)
                    if circ > 0.7: shape = 'circle'
                elif vtx == 3:
                    shape = 'triangle'
                elif vtx == 4:
                    x,y,wb,hb = cv2.boundingRect(cnt)
                    ar = float(wb)/hb
                    if 0.9 <= ar <= 1.1: shape = 'square'
                    elif ar > 1.3:       shape = 'horizontal_rectangle'
                    elif ar < 0.7:       shape = 'vertical_rectangle'
                    else:                shape = 'rectangle'

                if shape == self.target_shape:
                    x,y,wb,hb = cv2.boundingRect(cnt)
                    center_x = x + wb//2
                    center_y = y + hb//2
                    break

            # === คุมกิมบอลด้วย PID ไม่อิง ToF ===
            now = time.monotonic()
            dt  = max(1e-3, now - self.last_time)
            self.last_time = now

            cx, cy = w//2, h//2
            if center_x is not None and center_y is not None:
                have_object = True
                err_x = (center_x - cx)
                err_y = (center_y - cy)
                yaw_cmd   = pid_yaw.step(err_x, dt)
                pitch_cmd = pid_pitch.step(err_y, dt)

                # ขยับกิมบอลเล็กน้อยจนเข้าเป้า
                if abs(err_x) > 6 or abs(err_y) > 6:
                    self.ep_gimbal.drive_speed(pitch_speed=-pitch_cmd, yaw_speed=yaw_cmd)
                else:
                    # อยู่กลางพอดี -> ยิงและจบ
                    self.ep_gimbal.drive_speed(0,0)
                    self.ep_blaster.fire(times=3)
                    self.ep_gimbal.move(pitch=0, yaw=-15).wait_for_completed()
                    self.record_target_hit(self.target_color, self.target_shape)
                    break
            else:
                # ยังไม่เห็น/หลุดเป้า
                pid_yaw.integ *= 0.9
                pid_pitch.integ *= 0.9
                self.ep_gimbal.drive_speed(0,0)

            if time.monotonic() - t0 > timeout_s and not have_object:
                print("⏱️ หมดเวลาเล็ง (ยังไม่เห็นเป้า)")
                break

        self.ep_gimbal.drive_speed(0,0)
        
    @staticmethod
    def _infer_yaw_from_path(visited_path):
        """เดายอว์จากสองพอยต์สุดท้ายของ visited_path"""
        try:
            if len(visited_path) >= 2:
                (x1, y1), (x2, y2) = visited_path[-2], visited_path[-1]
                dx, dy = x2 - x1, y2 - y1
                if   (dx, dy) == (0, 1):  return 0      # North
                elif (dx, dy) == (1, 0):  return 90     # East
                elif (dx, dy) == (0,-1):  return 180    # South
                elif (dx, dy) == (-1,0):  return -90    # West
        except Exception:
            pass
        return 0  # เดาไม่ได้ ให้หันเหนือ

    def run_targets_shortest(self,
                           map_json_path: str,
                           targets_json_path: str,
                           target_filter: dict = None,
                           draw_after=True,
                           start_mode: str = "fresh"):  # "fresh" = (0,0, North), "resume" = จุดสิ้นสุดรอบก่อน
        """
        map_json_path: path ไปยัง map_state_run1_*.json
        targets_json_path: path ไปยัง targets_run1_*.json
        target_filter: เช่น {"color":"yellow","shape":"horizontal_rectangle"} (ถ้า None ใช้ทั้งหมด)
        """
        # โหลด JSON
        with open(map_json_path, "r", encoding="utf-8") as f:
            map_payload = json.load(f)
        with open(targets_json_path, "r", encoding="utf-8") as f:
            targets_payload = json.load(f)

        # กราฟที่สำรวจแล้ว
        graph = _graph_from_json(map_payload)
        self.internal_map.graph = {k: set(v) for k,v in graph.items()}
        self.internal_map.blocked = set(tuple(map(tuple, e)) for e in [
            (tuple(a), tuple(b)) for a,b in map_payload["blocked"]
        ])

        # ตั้งค่าเริ่มต้นตาม start_mode
        visited_path_prev = [(p[0], p[1]) for p in map_payload.get("visited_path", [])]
        
        if start_mode == "resume" and visited_path_prev:
            start_grid = visited_path_prev[-1]
            start_yaw = self._infer_yaw_from_path(visited_path_prev)
        else:
            start_grid = (0, 0)
            start_yaw = 0  # เหนือ

        # sync state
        self.current_position = tuple(start_grid)
        self.visited_path = [self.current_position]
        self.pose_handler.set_xy(start_grid[0]*GRID_SIZE_M, start_grid[1]*GRID_SIZE_M)
        self.pose_handler.set_yaw(float(start_yaw))

        # เลือกเป้าหมายที่ต้องไป
        all_targets = targets_payload.get("targets", [])
        if target_filter:
            color = target_filter.get("color")
            shape = target_filter.get("shape")
            use_targets = [t for t in all_targets
                           if (color is None or t["color"] == color)
                           and (shape is None or t["shape"] == shape)]
        else:
            use_targets = list(all_targets)

        if not use_targets:
            print("⚠️ ไม่พบเป้าหมายตามเงื่อนไขในไฟล์ targets")
            return

        # ตั้งค่าสี/รูปทรงที่ต้องเล็ง
        if target_filter and ("color" in target_filter or "shape" in target_filter):
            self.set_target(target_filter.get("color", self.target_color),
                            target_filter.get("shape", self.target_shape))
        else:
            # ถ้าไม่ได้ฟิลเตอร์ ให้ตั้งเป็นค่าสี/รูปทรงตัวแรก
            self.set_target(use_targets[0]["color"], use_targets[0]["shape"])

        # ไล่ไปทีละเป้าหมาย (เรียงใกล้สุดทีละจุดก็ได้)
        current = self.current_position
        remaining = [(t["grid"]["x"], t["grid"]["y"]) for t in use_targets]

        def manhattan(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

        while remaining:
            # เลือกจุดที่ใกล้ current ที่สุด (heuristic ง่ายๆ)
            remaining.sort(key=lambda p: manhattan(current, p))
            goal = remaining.pop(0)

            # หาเส้นทาง
            path = _bfs_path(graph, current, goal)
            if not path:
                print(f"❌ ไป {goal} ไม่ได้ (no path)")
                continue

            print(f"▶️ ไปยังเป้า {goal} ด้วย path: {path}")
            self.execute_path(path)        # ใช้ method เดิมจาก LAB 1
            current = goal
            self.current_position = goal   # sync
            self.pose_handler.set_xy(goal[0]*GRID_SIZE_M, goal[1]*GRID_SIZE_M)

            # ถึงแล้ว จอดนิ่ง -> เล็ง/ยิงแบบ no-ToF
            self.detect_and_shot_no_tof()

        print("✅ รอบ 2 เสร็จสิ้น")
        if draw_after:
            plot_map_with_walls(
                self.internal_map.graph,
                self.internal_map.blocked,
                self.visited_path,
                filename="maze_run2_path.png"
            )

# ================= MAIN =================
if __name__ == "__main__":
    print("Connecting to RoboMaster robot...")
    ep = robot.Robot()
    ep.initialize(conn_type="ap")
    print("Robot connected.")

    # สร้าง handler แบบ LAB 1 (ดึงจาก lab1)
    from test_lap1 import TofDataHandler, GimbalDataHandler, PoseDataHandler, PID

    tof = TofDataHandler()
    gim = GimbalDataHandler()
    pose = PoseDataHandler()

    # สมัคร sensor ที่จำเป็น (ตำแหน่ง/ท่าทาง/กิมบอล) — จะไม่พึ่ง ToF ตอนเล็งก็จริง
    ep.chassis.sub_position(freq=10, callback=pose.update_position)
    ep.chassis.sub_attitude(freq=10, callback=pose.update_attitude)
    ep.gimbal.sub_angle(freq=20, callback=gim.update)
    # (จะไม่ต้องอ่าน ToF ก็ได้ แต่ถ้าอยาก log ต่อ ก็สมัครไว้)
    ep.sensor.sub_distance(freq=10, callback=tof.update)
    START_MODE = "fresh"  # "fresh" = (0,0, North), "resume" = จุดสิ้นสุดรอบก่อน
    runner = MazeTargetRunner(ep, tof, gim, pose)
    runner.set_target('yellow', 'horizontal_rectangle')  # ค่า default เผื่อไม่มี filter

    # เริ่มที่ (0,0) yaw 0 ตาม LAB1
    pose.set_xy(0.0, 0.0)
    pose.set_yaw(0.0)

    try:
        # เลือกกรณี: ไปหาเฉพาะสี/รูปทรง หรือเอาทุกเป้าที่บันทึกไว้
        runner.run_targets_shortest(
            map_json_path=MAP_JSON_PATH,
            targets_json_path=TARGETS_JSON_PATH,
            target_filter={"color": "yellow", "shape": "horizontal_rectangle"},  # หรือ None
            draw_after=True,
            start_mode=START_MODE
        )
    finally:
        # cleanup เหมือน LAB 1
        try:
            ep.sensor.unsub_distance()
            ep.gimbal.unsub_angle()
            ep.chassis.unsub_position()
            ep.chassis.unsub_attitude()
            ep.close()
        except Exception as e:
            print("Cleanup warning:", e)
        print("Robot connection closed.")
    pose.set_yaw(0.0)

    try:
        # เลือกกรณี: ไปหาเฉพาะสี/รูปทรง หรือเอาทุกเป้าที่บันทึกไว้
        runner.run_targets_shortest(
            map_json_path=MAP_JSON_PATH,
            targets_json_path=TARGETS_JSON_PATH,
            target_filter={"color": "yellow", "shape": "horizontal_rectangle"},  # หรือ None
            draw_after=True,
            start_mode=START_MODE
        )
    finally:
        # cleanup เหมือน LAB 1
        try:
            ep.sensor.unsub_distance()
            ep.gimbal.unsub_angle()
            ep.chassis.unsub_position()
            ep.chassis.unsub_attitude()
            ep.close()
        except Exception as e:
            print("Cleanup warning:", e)
        print("Robot connection closed.")
