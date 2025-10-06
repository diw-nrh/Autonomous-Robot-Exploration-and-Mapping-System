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
import queue
from collections import deque
from datetime import datetime
import cv2
import numpy as np
# ==== ดึงของเดิมจาก LAB 1 ====
from lab1_complete_mul import (
    robot,                      # robomaster.robot
    MazeExplorer,               # คลาสหลักจาก LAB1
    GRID_SIZE_M,                # ขนาดกริด     # ฟังก์ชันวาดแผนที่ (เผื่ออยากบันทึก)
)
CAMERA_HORIZONTAL_FOV = 109
CAMERA_VERTICAL_FOV = 93
# ====== ตั้งชื่อไฟล์ JSON (แก้ให้ตรงกับไฟล์ที่ได้จาก LAB 1) ======
MAP_JSON_PATH = "map_state_run1_20251006_194731.json"
TARGETS_JSON_PATH = "targets_run1_20251006_194731.json"

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
    def execute_path(self, path, shoot_on_arrival=False, final_orientation: int = None):
        """
        เดินตามเส้นทางแบบไม่ยิงระหว่างทาง
        และปรับระยะจากกำแพงทุกครั้งหลังเดินแต่ละช่อง
        """
        if not path:
            return

        self.ep_led.set_led(r=0, g=0, b=255)

        for i in range(len(path) - 1):
            start_node, end_node = path[i], path[i + 1]
            dx, dy = end_node[0] - start_node[0], end_node[1] - start_node[1]

            # หาทิศและมุมเป้าหมาย
            if   (dx, dy) == (0, 1):  target_orientation, target_angle = 0,   0
            elif (dx, dy) == (1, 0):  target_orientation, target_angle = 1,  90
            elif (dx, dy) == (0,-1):  target_orientation, target_angle = 2, 180
            elif (dx, dy) == (-1,0):  target_orientation, target_angle = 3, -90
            else:
                continue  # path ผิดรูป

            # หมุนไปทิศที่ถูก
            self.turn_pid(target_angle)
            self.current_orientation = target_orientation
            time.sleep(0.5)
            # เดินหนึ่งช่อง
            print(self.tof_handler.get_distance())
            self.move_forward_pid(GRID_SIZE_M)
            time.sleep(0.5)
            # ✅ ปรับระยะห่างจากกำแพงเล็กน้อย
            try:
                self.periodic_wall_clearance_adjust(target_clearance_m=0.16)
            except Exception as e:
                print(f"[Adjust skipped] {e}")

            # อัปเดตตำแหน่ง/ทิศ
            self.current_position = end_node
            self.visited_path.append(self.current_position)
            self.pose_handler.set_xy(end_node[0] * GRID_SIZE_M, end_node[1] * GRID_SIZE_M)
            self.pose_handler.set_yaw(target_angle)
            time.sleep(0.2)

        # ✅ เมื่อถึงปลายทาง: จัดทิศรถตามที่ต้องการ (ถ้าระบุ)
        if final_orientation is not None:
            def _angle_from_orientation(o:int):
                return 0 if o == 0 else 90 if o == 1 else 180 if o == 2 else -90 if o == 3 else 0
            try:
                desired_angle = _angle_from_orientation(int(final_orientation))
            except Exception:
                desired_angle = 0
            try:
                print(f"↻ จัดทิศรถไปยัง orientation={final_orientation} (yaw={desired_angle}°)")
                self.turn_pid(desired_angle)
                self.current_orientation = int(final_orientation)
                self.pose_handler.set_yaw(float(desired_angle))
            except Exception as e:
                print(f"[Warn] ตั้งทิศตามเป้าหมายล้มเหลว: {e}")

        # ✅ ยิงเมื่อถึงเป้าหมาย (ถ้าระบุ)
        if shoot_on_arrival:
            try:
                self.ep_gimbal.moveto(yaw=0, pitch=-15).wait_for_completed()
            except Exception:
                pass
            self.detect_and_shot()

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
                           start_mode: str = "resume"):  # "fresh" = (0,0, North), "resume" = จุดสิ้นสุดรอบก่อน
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
        print(f"[LAB2] โหลด targets ทั้งหมด: {len(all_targets)}")
        if target_filter:
            def _norm(v):
                if v is None:
                    return None
                if isinstance(v, str) and v.strip().lower() in ("none", "", "*", "any", "all"):
                    return None
                return v
            color = _norm(target_filter.get("color"))
            shape = _norm(target_filter.get("shape"))
            print(f"[LAB2] ใช้ตัวกรอง -> color: {color if color is not None else '*'}, shape: {shape if shape is not None else '*'}")
            use_targets = [t for t in all_targets
                           if (color is None or t["color"] == color)
                           and (shape is None or t["shape"] == shape)]
        else:
            print("[LAB2] ไม่ได้ระบุตัวกรอง ใช้ทุกเป้าหมาย")
            use_targets = list(all_targets)

        if not use_targets:
            print("⚠️ ไม่พบเป้าหมายตามเงื่อนไขในไฟล์ targets")
            if all_targets:
                print("   เป้าหมายที่มีอยู่:")
                for i, t in enumerate(all_targets, 1):
                    print(f"   - {i}. {t.get('color','?')} {t.get('shape','?')} @ grid=({t.get('grid',{}).get('x','?')},{t.get('grid',{}).get('y','?')})")
            return

        # ตั้งค่าสี/รูปทรงที่ต้องเล็ง
        if target_filter and ("color" in target_filter or "shape" in target_filter):
            self.set_target(target_filter.get("color", self.target_color),
                            target_filter.get("shape", self.target_shape))
        else:
            # ถ้าไม่ได้ฟิลเตอร์ ให้ตั้งเป็นค่าสี/รูปทรงตัวแรก
            self.set_target(use_targets[0]["color"], use_targets[0]["shape"])

        # ไล่ไปทีละ "เป้าหมาย" (grid + orientation) เพื่อคงทิศตามไฟล์ targets
        current = self.current_position
        remaining_targets = list(use_targets)
        self.ep_camera.start_video_stream(display=False, resolution="720p")
        def manhattan(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

        while remaining_targets:
            # เลือกเป้าหมายที่กริดใกล้ current ที่สุด
            remaining_targets.sort(key=lambda t: manhattan(current, (t["grid"]["x"], t["grid"]["y"])) )
            tgt = remaining_targets.pop(0)

            goal = (tgt["grid"]["x"], tgt["grid"]["y"])
            print(graph, current, goal)
            path = _bfs_path(graph, current, goal)
            if not path:
                print(f"❌ ไป {goal} ไม่ได้ (no path)")
                continue

            # ตั้งสี/รูปทรง แล้วไปถึงกริดปลายทาง จากนั้นจัดทิศตาม target นี้และยิง
            self.set_target(tgt["color"], tgt["shape"])
            fin_orient = tgt.get("orientation")
            self.execute_path(path, shoot_on_arrival=True, final_orientation=fin_orient)

            current = goal
            self.current_position = goal
            self.pose_handler.set_xy(goal[0] * GRID_SIZE_M, goal[1] * GRID_SIZE_M)
        

        print("✅ รอบ 2 เสร็จสิ้น")
        if draw_after:
            # ใช้ signature แบบอ้างอิงสถานะภายในอ็อบเจ็กต์เท่านั้น
            self.plot_map_with_walls(filename="maze_run2_path.png")
        self.ep_camera.stop_video_stream()

# ================= MAIN =================
if __name__ == "__main__":
    print("Connecting to RoboMaster robot...")
    ep = robot.Robot()
    ep.initialize(conn_type="ap")
    print("Robot connected.")

    # สร้าง handler แบบ LAB 1 (ดึงจาก lab1)
    from lab1_complete_mul import TofDataHandler, GimbalDataHandler, PoseDataHandler, PID

    tof = TofDataHandler()
    gim = GimbalDataHandler()
    pose = PoseDataHandler()

    # สมัคร sensor ที่จำเป็น (ตำแหน่ง/ท่าทาง/กิมบอล) — จะไม่พึ่ง ToF ตอนเล็งก็จริง
    ep.chassis.sub_position(freq=10, callback=pose.update_position)
    ep.chassis.sub_attitude(freq=10, callback=pose.update_attitude)
    ep.gimbal.sub_angle(freq=20, callback=gim.update)
    # (จะไม่ต้องอ่าน ToF ก็ได้ แต่ถ้าอยาก log ต่อ ก็สมัครไว้)
    ep.sensor.sub_distance(freq=10, callback=tof.update)
    START_MODE = "resume"  # "fresh" = (0,0, North), "resume" = จุดสิ้นสุดรอบก่อน
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
            target_filter=None,  # หรือกำหนดเฉพาะเจาะจง เช่น {"color": "green", "shape": "square"}
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