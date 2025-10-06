SHOW_DEBUG = False
"""
=================================================================================
LAB 1: MAZE EXPLORATION ROBOT
=================================================================================
Description: หุ่นยนต์สำรวจเขาวงกต โดยใช้ ToF sensor ด้านหน้า และ IR sensors ด้านข้าง
            เฉพาะการสำรวจทั้งหมด ไม่มี goal

Features:
- ToF sensor สำหรับวัดระยะด้านหน้า
- IR digital sensors สำหรับตรวจจับกำแพงด้านซ้าย/ขวา
- Real-time mapping แสดงแผนที่ขณะสำรวจ
- Complete maze exploration only
- CSV logging และ state saving/loading

Hardware Requirements:
- RoboMaster EP/S1
- ToF sensor (built-in)
- IR digital sensors x2 (port 1: ซ้าย, port 2: ขวา)

Author: [Your Name]
Date: October 2025
=================================================================================
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================
from robomaster import blaster
import time
import threading
import math
import csv
from datetime import datetime
from collections import deque
import json
import os
import sys
import statistics
import cv2
import robomaster
from robomaster import robot, vision
import queue
# Matplotlib for real-time plotting and map visualization
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Robot Movement and Grid Settings
GRID_SIZE_M = 0.6                   # ขนาดกริดเป็นเมตร
WALL_THRESHOLD_MM = 450                # เกณฑ์ระยะทางที่ถือว่ามีกำแพง (มม.)

# Gimbal and Vision Settings
GIMBAL_TURN_SPEED = 360                # ความเร็วหมุนกิมบอล (องศา/วินาที)
CAMERA_HORIZONTAL_FOV = 109
CAMERA_VERTICAL_FOV = 93
VISION_SCAN_DURATION_S = 1             # ระยะเวลาสแกน vision (วินาที)

# Direction Mappings
ORIENTATIONS = {
    0: "North", 
    1: "East", 
    2: "South", 
    3: "West"
}

WALL_NAMES = {
    0: "North Wall", 
    1: "East Wall", 
    2: "South Wall", 
    3: "West Wall"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def wrap_angle_deg(angle_deg: float) -> float:
    """ห่อมุมให้อยู่ในช่วง (-180, 180]"""
    while angle_deg <= -180.0:
        angle_deg += 360.0
    while angle_deg > 180.0:
        angle_deg -= 360.0
    return angle_deg

def sub_angle(a_deg: float, b_deg: float) -> float:
    """คืนค่า (a - b) แบบ wrap ในช่วง (-180, 180]"""
    return wrap_angle_deg(a_deg - b_deg)


# =============================================================================
# DATA HANDLERS AND SENSOR MANAGEMENT
# =============================================================================
class PID:
    def __init__(self, kp=0.2, ki=0.0, kd=0.0, integ_clip=40.0, out_clip=10.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integ = 0.0
        self.prev_err = 0.0
        self.integ_clip = float(integ_clip)
        self.out_clip = float(out_clip)
    def reset(self):
        self.integ = 0.0; self.prev_err = 0.0
    def step(self, err, dt):
        dt = max(dt, 1e-3)
        self.integ += err * dt
        self.integ = max(-self.integ_clip, min(self.integ, self.integ_clip))
        deriv = (err - self.prev_err) / dt
        self.prev_err = err
        out = self.kp*err + self.ki*self.integ + self.kd*deriv
        return max(-self.out_clip, min(self.out_clip, out))


                
class TofDataHandler:
    """จัดการข้อมูลจาก ToF sensor ด้วย median filtering"""
    
    def __init__(self, window_size=3):
        self._lock = threading.Lock()
        self.raw_distance = 0.0
        self._window = deque(maxlen=window_size)
        self._median = 0.0

    def update(self, sub_info):
        """อัปเดตค่า ToF จาก callback"""
        d = float(sub_info[0]) if sub_info else 0.0
        with self._lock:
            self.raw_distance = d
            self._window.append(d)
            self._median = statistics.median(self._window)

    def get_distance(self):
        """คืนค่าระยะทางที่กรองแล้ว (median)"""
        with self._lock:
            return self._median if self._window else self.raw_distance

    def get_raw_distance(self):
        with self._lock:
            return self.raw_distance


class GimbalDataHandler:
    """จัดการข้อมูลมุมกิมบอล (yaw, pitch)"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.yaw = 0.0
        self.pitch = 0.0

    def update(self, angle_info):
        """อัปเดตมุมกิมบอลจาก callback"""
        with self._lock:
            self.yaw = float(angle_info[0])
            self.pitch = float(angle_info[1])

    def get_yaw_pitch(self):
        """คืนค่ามุม (yaw, pitch)"""
        with self._lock:
            return (self.yaw, self.pitch)


class PoseDataHandler:
    """จัดการข้อมูลตำแหน่งและทิศทางของหุ่นยนต์"""
    
    def __init__(self):
        self.pose = [0.0] * 6  # [x, y, z, yaw, pitch, roll]
        self._lock = threading.Lock()
        
    def update_position(self, pos_info):
        """อัปเดตตำแหน่ง (x, y, z)"""
        with self._lock:
            self.pose[0], self.pose[1], self.pose[2] = pos_info[0], pos_info[1], pos_info[2]
            
    def update_attitude(self, att_info):
        """อัปเดตทิศทาง (yaw, pitch, roll)"""
        with self._lock:
            self.pose[3], self.pose[4], self.pose[5] = att_info[0], att_info[1], att_info[2]
            
    def get_pose(self):
        """คืนค่า pose ทั้งหมด"""
        with self._lock:
            return tuple(self.pose)
            
    def set_xy(self, x_m, y_m):
        """กำหนดตำแหน่ง x, y"""
        with self._lock:
            self.pose[0], self.pose[1] = float(x_m), float(y_m)
            
    def set_yaw(self, yaw_deg):
        """กำหนดทิศทาง yaw"""
        with self._lock:
            self.pose[3] = float(yaw_deg)


# =============================================================================
# MAP AND NAVIGATION CLASSES
# =============================================================================

class RobotMap:
    """จัดการแผนที่และเส้นทางของหุ่นยนต์"""
    
    def __init__(self):
        self.graph = {}           # กราฟเชื่อมต่อระหว่างช่อง
        self.explored = set()     # ช่องที่สำรวจแล้ว
        self.blocked = set()      # ขอบที่ปิดกั้น (กำแพง)
        
    def add_connection(self, pos1, pos2):
        """เพิ่มการเชื่อมต่อระหว่างสองช่อง"""
        if pos1 not in self.graph: 
            self.graph[pos1] = set()
        if pos2 not in self.graph: 
            self.graph[pos2] = set()
        if pos2 not in self.graph[pos1]:
            self.graph[pos1].add(pos2)
            self.graph[pos2].add(pos1)

    def add_blocked(self, pos1, pos2):
        """เพิ่มขอบที่ปิดกั้น (กำแพง)"""
        edge = tuple(sorted([pos1, pos2]))
        self.blocked.add(edge)
        
    def mark_explored(self, position):
        """ทำเครื่องหมายช่องที่สำรวจแล้ว"""
        self.explored.add(position)
        
    def get_unexplored_neighbors(self, position):
        """หาช่องใกล้เคียงที่ยังไม่ได้สำรวจ"""
        if position not in self.graph: 
            return []
        return [n for n in self.graph.get(position, []) if n not in self.explored]
        
    def get_path(self, start, goal):
        """หาเส้นทางจาก start ไป goal ด้วย BFS"""
        if start == goal: 
            return [start]
        queue = [(start, [start])]
        visited = {start}
        print(f"Finding path from {start} to {goal}")
        while queue:
            current, path = queue.pop(0)
            for neighbor in self.graph.get(current, []):
                if neighbor not in visited:
                    if neighbor == goal: 
                        return path + [neighbor]
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None



class PIDController:
    """ตัวควบคุม PID สำหรับการเคลื่อนที่"""
    
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(-1.0, 1.0)):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self._integral = 0.0
        self._previous_error = 0.0
        self._last_time = time.time()
        
    def update(self, current_value):
        """อัปเดตผลลัพธ์ PID"""
        current_time = time.time()
        dt = current_time - self._last_time
        if dt <= 0: 
            return 0.0
        error = self.setpoint - current_value
        self._integral += error * dt
        derivative = (error - self._previous_error) / dt
        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)
        if self.output_limits:
            output = max(self.output_limits[0], min(self.output_limits[1], output))
        self._previous_error = error
        self._last_time = current_time
        return output


# =============================================================================
# MAIN MAZE EXPLORER CLASS
# =============================================================================

class MazeExplorer:
    """
    คลาสหลักสำหรับควบคุมหุ่นยนต์สำรวจเขาวงกต
    
    Features:
    - สำรวจเขาวงกตแบบอัตโนมัติ
    - ใช้ ToF + IR sensors
    - Real-time mapping
    - Complete exploration only
    - CSV logging
    """
    
    def __init__(self, ep_robot, tof_handler, gimbal_handler, pose_handler):
        """
        Initialize MazeExplorer
        
        Args:
            ep_robot: RoboMaster robot instance
            tof_handler: ToF sensor data handler
            vision_handler: Vision/marker detection handler
            gimbal_handler: Gimbal angle handler
            pose_handler: Robot pose handler
        """
        # Robot hardware interfaces
        self.ep_robot = ep_robot
        self.ep_chassis = ep_robot.chassis
        self.ep_led = ep_robot.led
        self.ep_gimbal = ep_robot.gimbal
        self.ep_adaptor = ep_robot.sensor_adaptor
        self.ep_camera = ep_robot.camera
        self.blaster = blaster
        self.ep_blaster = ep_robot.blaster
        # Data handlers
        self.tof_handler = tof_handler
        self.gimbal_handler = gimbal_handler
        self.pose_handler = pose_handler
        self.last_time = time.monotonic()
        # Navigation state
        self.shoot_grid = [] 
        self.visited_for_check = []
        self.current_position = (0, 0)
        self.current_orientation = 0  # 0:N, 1:E, 2:S, 3:W
        self.internal_map = RobotMap()
        self.visited_path = [self.current_position]
        self.count = True
        # Exploration settings
        self.border = (3, 3)  # ขนาดพื้นที่สำรวจ (กริด)
        self.step_counter = 0  # นับจำนวนช่องที่เดิน
        
        # Exploration settings
        self.exploration_complete = False  # flag สำรวจครบหรือยัง
        
        # Marker detection (ถ้าต้องการใช้)
        self.marker_map = {}  # {name: [{grid_pos:(x,y), wall:str, offset_m:float}]}
        
        # Real-time plotting
        self.real_time_plot = True
        self.fig = None
        self.ax = None
        
        # Initialize robot
        self.ep_led.set_led(r=0, g=0, b=0)
        self.pose_handler.set_xy(0.0, 0.0)
        self.pose_handler.set_yaw(0.0)
        
        # Logging
        self.scan_log = []      # สแกนแต่ละกริด
        self.wall_log = set()   # กำแพงที่พบ
        self.path_log = []      # เส้นทางที่เดิน
        self.continuous_sensor_log = []  # บันทึกเซ็นเซอร์ต่อเนื่อง
        self.watched_grid = []    # บันทึกจุดที่ยิง
        self.detected_targets = []        # list ของเป้าหมายที่เจอ/ยิง
        self.targets_json_path = None    # path ไฟล์ targets_*.json
        self.map_state_json_path = None   # path ไฟล์ map_state_*.json

        # Logging thread
        self._logging_thread = None
        self._logging_thread_active = False
       
        self.ep_robot = ep_robot
        self.ep_camera = ep_robot.camera
        self.ep_gimbal = ep_robot.gimbal
        self.ep_blaster = ep_robot.blaster
        self.last_time = time.monotonic()
        
        # Color ranges (HSV)
        self.color_ranges = {
            # red ต้อง 2 ช่วง: 0–25 และ 170–180 (tuple ของคู่ช่วงสี)
            'brown_red': (([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])),
            'dark_green': ([40, 60, 20],   [85, 255, 180]),
            'yellow_brown': ([18, 100, 100],[40, 255, 255]),
            'dark_blue': ([95, 70, 20],    [130, 255, 200])
        }
        
        # Color mapping
        self.colors = {
            'red': "brown_red",
            'green': "dark_green", 
            'yellow': "yellow_brown",
            'blue': "dark_blue",
        }
        
        self.target_color = None
        self.target_shape = None
    
    def set_target(self, color, shape):
        """ตั้งค่าเป้าหมาย"""
        self.target_color = color
        self.target_shape = shape
    # === Helpers: สี→mask, จัดประเภททรง, อัปเดต shoot_grid ===
    def _build_mask(self, hsv_img, color_data):
        """รองรับทั้งกรณีช่วงสีเดี่ยว และกรณีสีแดงแบบ 2 ช่วง"""
        is_dual = (
            isinstance(color_data, tuple)
            and len(color_data) == 2
            and all(isinstance(r, (list, tuple, np.ndarray)) and len(r) == 2 for r in color_data)
        )
        if is_dual:
            (l1, u1), (l2, u2) = color_data
            m1 = cv2.inRange(hsv_img, np.array(l1, np.uint8), np.array(u1, np.uint8))
            m2 = cv2.inRange(hsv_img, np.array(l2, np.uint8), np.array(u2, np.uint8))
            return cv2.bitwise_or(m1, m2)
        lb = np.array(color_data[0], np.uint8)
        ub = np.array(color_data[1], np.uint8)
        return cv2.inRange(hsv_img, lb, ub)

    def _classify_shape(self, cnt):
        """คืนค่า: 'circle' | 'square' | 'horizontal_rectangle' | 'vertical_rectangle' | None"""
        area = cv2.contourArea(cnt)
        if area < 150:
            return None
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * max(1.0, peri), True)
        v = len(approx)
        if v == 3:
            return None  # ไม่ใช้สามเหลี่ยมในงานนี้
        if v == 4:
            x, y, wb, hb = cv2.boundingRect(cnt)
            ar = float(wb) / max(1.0, hb)
            if 0.85 <= ar <= 1.15:
                return 'square'
            elif ar > 1.15:
                return 'horizontal_rectangle'
            else:
                return 'vertical_rectangle'
        if v >= 8:
            circ = 4.0 * np.pi * area / max(peri * peri, 1e-6)
            if circ > 0.7:
                return 'circle'
        return None

    def _update_shoot_grid(self, pos, ori, detections, replace=True):
        """
        detections: list of (color, shape, (cx,cy))
        เก็บเป็นรูปแบบใหม่: (pos, ori, [(color,shape), ...])
        replace=True → เขียนทับผลเดิมของกริด/ทิศนั้นด้วยผลสแกนรอบนี้ (แม้จะว่าง)
        """
        
        if not isinstance(self.shoot_grid, list):
            self.shoot_grid = []
        entry_idx = None
        for i, entry in enumerate(self.shoot_grid):
            # migrate โครงเก่า → ใหม่
            if len(entry) == 4 and isinstance(entry[2], str):
                p, o, c_old, s_old = entry
                self.shoot_grid[i] = (p, o, [(c_old, s_old)])
                entry = self.shoot_grid[i]
            if len(entry) == 3 and entry[0] == pos and entry[1] == ori:
                entry_idx = i
                break
        if entry_idx is None:
            self.shoot_grid.append((pos, ori, []))
            entry_idx = len(self.shoot_grid) - 1

        p, o, cs_list = self.shoot_grid[entry_idx]
        print(1,cs_list)
        if replace:
            # ใช้ผลสแกน “รอบนี้เท่านั้น”
            cs_list = [(c, s) for (c, s, _center) in detections]
        else:
            # สะสม (โหมดเดิม)
            for c, s, _center in detections:
                cs_list.append((c, s))
        print(2,cs_list)
        self.shoot_grid[entry_idx] = (p, o, cs_list)
        return cs_list  # คืนรายการ (color,shape) ของกริดนี้
    def _save_detection_snapshot(self, img_bgr, boxes_abs, labels, out_dir="snapshots", tag="phase1", roi_rect=None):
        """
        บันทึกภาพเต็ม พร้อมกรอบวัตถุ (boxes_abs ในพิกัดเต็มภาพ) และกรอบ ROI (ถ้ามี)
        """
        try:
            os.makedirs(out_dir, exist_ok=True)
            vis = img_bgr.copy()

            # วาดกรอบ ROI (ถ้ามี)
            if roi_rect is not None:
                rx, ry, rw, rh = roi_rect
                cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
                cv2.putText(vis, "ROI", (rx, max(0, ry - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # วาดกรอบวัตถุ
            for (bx, by, bw, bh), text in zip(boxes_abs, labels):
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(vis, text, (bx, max(0, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            gx, gy = self.current_position
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fname = f"{out_dir}/snap_{tag}_g{gx}_{gy}_o{self.current_orientation}_{ts}.png"
            cv2.imwrite(fname, vis)
            print(f"[SNAPSHOT] saved → {fname}")
        except Exception as e:
            print(f"[SNAPSHOT] error: {e}")
            
    def scan_objects_fullframe(self, store=True, save_snapshot=True):
        """
        สแกนภาพเต็มเฟรม 1 ช็อต → คืน list[(user_color, shape, (cx,cy))]
        ถ้า store=True จะอัปเดต self.shoot_grid ของ (pos, ori) ปัจจุบันด้วย
        ถ้า save_snapshot=True จะบันทึกภาพพร้อมกรอบ
        """
        # ลอง recenter ให้ภาพอยู่มุมกลางก่อนสแกน (กันค้าง)
        self.ep_gimbal.moveto(pitch=-10, yaw=0, pitch_speed=120).wait_for_completed()
        img_full = None
        for _ in range(4):
            img_full = self.ep_camera.read_cv2_image(strategy="newest")
            time.sleep(0.5)
        detections = []
        boxes_abs = []
        labels = []
        roi_rect = None  # <— เตรียมไว้ส่งให้ snapshot
        if img_full is not None:
            # บังคับเรียงจากซ้ายภาพไปขวา
            detections.sort(key=lambda d: d[2][0])
            h, w = img_full.shape[:2]
            roi_w, roi_h = min(550, w), min(300, h)
            roi_x, roi_y = (w - roi_w) // 2, (h - roi_h) // 2
            roi_rect = (roi_x, roi_y, roi_w, roi_h)  # <— เก็บพิกัด ROI (absolute)
            roi = img_full[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            blur_full = cv2.GaussianBlur(roi, (5, 5), 0)
            hsv_full = cv2.cvtColor(blur_full, cv2.COLOR_BGR2HSV)

            reverse_color = {v: k for k, v in self.colors.items()} if hasattr(self, 'colors') else {}

            for color_key, color_data in self.color_ranges.items():
                mask = self._build_mask(hsv_full, color_data)
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    shp = self._classify_shape(cnt)
                    if not shp:
                        continue
                    x, y, wb, hb = cv2.boundingRect(cnt)
                    # ศูนย์กลางในพิกัด ROI
                    cx, cy = x + wb // 2, y + hb // 2
                    # แปลงเป็นพิกัดเต็มภาพ
                    abs_x, abs_y = roi_x + x, roi_y + y
                    user_color = reverse_color.get(color_key, color_key)  # 'brown_red' → 'red'
                    detections.append((user_color, shp, (roi_x + cx, roi_y + cy - 20)))
                    boxes_abs.append((abs_x, abs_y, wb, hb))
                    labels.append(f"{user_color} {shp}")
        # บังคับเรียงซ้าย→ขวา (tie-break ด้วย y เพื่อความนิ่ง)
        if detections:
            detections.sort(key=lambda d: (d[2][0], d[2][1]))
        print(detections)
        if store:
            pos = self.current_position
            ori = self.current_orientation
            cs_list = self._update_shoot_grid(pos, ori, detections, replace=True)
            print(f"[SCAN] pos={pos} ori={ori} → {cs_list}")

        if save_snapshot and img_full is not None:
            self._save_detection_snapshot(
                img_full, boxes_abs, labels,
                out_dir="snapshots", tag="phase1", roi_rect=roi_rect
            )

        return detections

    def detect_and_shot(self):
        """
        Phase 2: เล็ง/ยิง เฉพาะเมื่อ (target_color, target_shape) มีอยู่ในกริดปัจจุบัน
        """
        self.ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
        tof_dist = self.tof_handler.get_distance()
        if 300 < tof_dist < 1200 and self.current_position and (self.current_position, self.current_orientation) not in self.watched_grid:
            self.watched_grid.append((self.current_position, self.current_orientation))
            # Phase 1 (บันทึกภาพด้วย)
            detections = self.scan_objects_fullframe(store=True, save_snapshot=True)
            time.sleep(0.5)
            pairs = [(c, s) for (c, s, _center) in detections]
            center_pairs = [ctr for (c, s, ctr) in detections
                            if (c, s) == (self.target_color, self.target_shape)]
            # เรียงซ้าย→ขวา
            center_pairs.sort(key=lambda p: p[0])
            pos, ori = self.current_position, self.current_orientation

            if (self.target_color, self.target_shape) not in pairs:
                print(f"⚠️ ไม่พบ {self.target_color}-{self.target_shape} ที่ {pos} ori={ori}; ข้ามการเล็งยิง")
                return None

            # 3) เริ่มเล็ง/ยิง (ของเดิม) — ย่อให้เหลือแก่น
            pid_yaw = PID(kp=0.20, ki=0.00, kd=0.00, integ_clip=40.0, out_clip=10.0)
            pid_pitch = PID(kp=0.20, ki=0.00, kd=0.00, integ_clip=40.0, out_clip=8.0)
            try:
                self.ep_gimbal.moveto(pitch=-10, yaw=0, pitch_speed=120).wait_for_completed()
            except Exception:
                pass
            time.sleep(0.2)

            center_x = center_y = None
            timebefore = time.monotonic()
            self.last_time = time.monotonic()
            for i in range(len(center_pairs)):
                center_x, center_y = center_pairs[i]
                dist_tof = self.tof_handler.get_distance()
                # 1) แปลงพิกัดพิกเซล → มุม (ใช้ FOV)
                # ใช้ขนาดภาพจริงถ้ามี; ถ้าไม่มีตั้งเป็น 1280x720
                w, h = 1280.0, 720.0
                hfov = math.radians(CAMERA_HORIZONTAL_FOV)
                vfov = math.radians(CAMERA_VERTICAL_FOV)
                fx = (w / 2.0) / math.tan(hfov / 2.0)   # focal length (px) แนวนอน
                fy = (h / 2.0) / math.tan(vfov / 2.0)   # focal length (px) แนวตั้ง
                dx = center_x - (w / 2.0)
                dy = center_y - (h / 2.0)
                yaw_err_deg   = math.degrees(math.atan2(dx, fx))     # + คือหันขวา
                pitch_err_deg = -math.degrees(math.atan2(dy, fy))    # + คือเงยขึ้น (y จอเพิ่มลง)

                # 2) สเกลตามระยะ ToF: ใกล้มาก → ก้าวสั้นลง
                #    300–1000 mm → scale ~ 0.4–1.0 (ปรับได้ตามหน้างาน)
                scale = max(0.4, min(1.0, (dist_tof - 300.0) / 700.0))
                yaw_step   = max(-25.0, min(25.0,  yaw_err_deg   * scale))
                pitch_step = max(-20.0, min(20.0,  pitch_err_deg * scale))

                # 3) ขยับแบบ relative เพื่อเลี่ยง absolute jump
                try:
                    self.ep_gimbal.move(yaw=yaw_step, pitch=pitch_step-5, yaw_speed=60, pitch_speed=60).wait_for_completed()
                except Exception:
                    # fallback: ใช้ moveto โดยบวกเพิ่มจากมุมปัจจุบัน
                    cyaw, cpitch = self.gimbal_handler.get_yaw_pitch()
                    self.ep_gimbal.moveto(yaw=cyaw + yaw_step, pitch=cpitch + pitch_step,
                                          yaw_speed=60, pitch_speed=60).wait_for_completed()
                while True:
                    try:
                        img = self.ep_camera.read_cv2_image(strategy="newest", timeout=3)
                        if img is None:
                            continue
                    except queue.Empty:
                        continue

                    color_key = self.colors.get(self.target_color, self.target_color)
                    if color_key not in self.color_ranges:
                        print(f"⚠️ ไม่พบสี '{self.target_color}' ในระบบ")
                        return []

                    # ROI กลางภาพ
                    h, w = img.shape[:2]
                    roi_w, roi_h = min(250, w), min(250, h)
                    roi_x, roi_y = (w - roi_w) // 2, (h - roi_h) // 2
                    roi = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

                    blur = cv2.GaussianBlur(roi, (5, 5), 0)
                    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
                    mask = self._build_mask(hsv, self.color_ranges[color_key])
                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    target_found = False
                    for cnt in contours:
                        if cv2.contourArea(cnt) < 200:
                            continue
                        shape = self._classify_shape(cnt)
                        if shape == self.target_shape:
                            x, y, wb, hb = cv2.boundingRect(cnt)
                            center_x = roi_x + x + wb // 2
                            center_y = roi_y + y + hb // 2 - 20
                            target_found = True
                            break

                    # แสดงหน้าต่างเพื่อดีบัก (ถ้าต้องการคงไว้)
                    try:
                        cv2.imshow("RoboMaster Detection", img)
                        cv2.imshow("Mask", mask)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("🔴 User pressed 'q' - exiting detection")
                            break
                    except Exception:
                        pass

                    now = time.monotonic()
                    dt = now - self.last_time
                    self.last_time = now

                    camera_w = 1280 // 2
                    camera_h = 720 // 2

                    if target_found:
                        err_x = (center_x - camera_w)
                        err_y = (center_y - camera_h)
                        yaw_cmd = pid_yaw.step(err_x, dt)
                        pitch_cmd = pid_pitch.step(err_y, dt)
                        if abs(yaw_cmd) > 0.05 or abs(pitch_cmd) > 0.05:
                            try:
                                self.ep_gimbal.drive_speed(pitch_speed=-pitch_cmd, yaw_speed=yaw_cmd)
                            except Exception:
                                try:
                                    self.ep_gimbal.drive_speed(pitch_speed=pitch_cmd, yaw_speed=yaw_cmd)
                                except Exception:
                                    pass
                        else:
                            print("🎯 Target centered - firing and exiting")
                            self.ep_blaster.fire(fire_type=self.blaster.INFRARED_FIRE,times=3)
                            time.sleep(0.8)
                            self.ep_blaster.fire(fire_type=self.blaster.INFRARED_FIRE,times=3)
                            time.sleep(0.8)
                            self.ep_blaster.fire(fire_type=self.blaster.INFRARED_FIRE,times=3)
                            time.sleep(0.6)
                            self.record_target_hit(self.target_color, self.target_shape)
                            try:
                                self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                                self.ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
                            except Exception:
                                pass
                            try:
                                cv2.destroyAllWindows()
                            except Exception:
                                pass
                            break

                    elif time.monotonic() - timebefore > 10.0:
                        # กันวนค้างนานเกินถ้าไม่เจอ
                        try:
                            self.ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
                            cv2.destroyAllWindows()
                        except Exception:
                            pass
                        break

                    center_x = center_y = None
                self.ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
            # end detect_and_shot
    # -------------------- Continuous Sensor Logging (NEW) --------------------
    def _continuous_log_worker(self, frequency_hz=10):
        print(f"[Logging Thread] Started. Logging at {frequency_hz} Hz.")
        period = 1.0 / max(1, frequency_hz)
        while self._logging_thread_active:
            t0 = time.time()
            try:
                x, y, z, yaw, pitch, roll = self.pose_handler.get_pose()
                tof = self.tof_handler.get_distance()
            
                self.continuous_sensor_log.append({
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "pos_x_m": x, "pos_y_m": y, "pos_z_m": z,
                    "att_yaw_deg": yaw, "att_pitch_deg": pitch, "att_roll_deg": roll,
                    "tof_distance_mm": tof,
                })
            except Exception as e:
                print(f"[Logging Thread] Error: {e}")
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
        print("[Logging Thread] Stopped.")

    def start_continuous_logging(self, frequency_hz=10):
        if not self._logging_thread_active:
            self._logging_thread_active = True
            self._logging_thread = threading.Thread(
                target=self._continuous_log_worker, kwargs={"frequency_hz": frequency_hz}, daemon=True
            )
            self._logging_thread.start()

    def stop_continuous_logging(self):
        if self._logging_thread_active:
            self._logging_thread_active = False
            if self._logging_thread:
                self._logging_thread.join(timeout=2.0)

    # ------------------------------------------------------------------------
    def save_csv_logs(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1) continuous sensor log
        if self.continuous_sensor_log:
            try:
                with open(f"continuous_sensor_log_{ts}.csv", "w", newline='', encoding="utf-8") as f:
                    fieldnames = ["timestamp", "pos_x_m", "pos_y_m", "pos_z_m",
                                  "att_yaw_deg", "att_pitch_deg", "att_roll_deg",
                                  "tof_distance_mm"]
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(self.continuous_sensor_log)
                print(f"[CSV] continuous_sensor_log_{ts}.csv saved")
            except Exception as e:
                print(f"[CSV] Error saving continuous log: {e}")

        # 2) scan per grid
        if self.scan_log:
            with open(f"scan_log_{ts}.csv", "w", newline='', encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["ts","grid_x","grid_y","N_mm","E_mm","S_mm","W_mm"])
                w.writeheader()
                for r in self.scan_log: w.writerow(r)

        # 3) closed walls
        if self.wall_log:
            with open(f"walls_log_{ts}.csv", "w", newline='', encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["cell1_x","cell1_y","cell2_x","cell2_y"])
                for (a,b) in sorted(self.wall_log):
                    w.writerow([a[0],a[1],b[0],b[1]])


        # 5) visited path
        if self.visited_path:
            with open(f"path_log_{ts}.csv", "w", newline='', encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["step","grid_x","grid_y"])
                for i, (gx,gy) in enumerate(self.visited_path):
                    w.writerow([i,gx,gy])
    def adc_to_distance_left(self, adc_value):
        """
        แปลงค่า ADC เป็น mm สำหรับ Sharp Left
        ข้อมูลการวัด:
        150mm -> 224 ADC
        100mm -> 357 ADC
        50mm  -> 698 ADC
        
        ใช้ linear interpolation และ extrapolation
        """
        if adc_value <= 0:
            return 999  # Invalid reading
        
        # ข้อมูลจุดอ้างอิง (adc, distance)
        points = [(224, 150), (357, 100), (698, 50)]
        
        # หาจุดที่ใกล้ที่สุด 2 จุดสำหรับ interpolation
        if adc_value <= 224:
            # Extrapolate จากจุดแรกสองจุด
            x1, y1 = 224, 150
            x2, y2 = 357, 100
        elif adc_value <= 357:
            # Interpolate ระหว่างจุดแรกและจุดที่สอง
            x1, y1 = 224, 150
            x2, y2 = 357, 100
        elif adc_value <= 698:
            # Interpolate ระหว่างจุดที่สองและจุดที่สาม
            x1, y1 = 357, 100
            x2, y2 = 698, 50
        else:
            # Extrapolate จากจุดสุดท้ายสองจุด
            x1, y1 = 357, 100
            x2, y2 = 698, 50
        
        # Linear interpolation: y = y1 + (y2-y1) * (x-x1) / (x2-x1)
        distance = y1 + (y2 - y1) * (adc_value - x1) / (x2 - x1)
        
        return max(20, min(300, distance))

    def adc_to_distance_right(self, adc_value):
        """
        แปลงค่า ADC เป็น mm สำหรับ Sharp Right
        ข้อมูลการวัด:
        150mm -> 235 ADC
        100mm -> 364 ADC
        50mm  -> 705 ADC
        
        ใช้ linear interpolation และ extrapolation
        """
        if adc_value <= 0:
            return 999  # Invalid reading
        
        # ข้อมูลจุดอ้างอิง (adc, distance)
        points = [(235, 150), (364, 100), (705, 50)]
        
        # หาจุดที่ใกล้ที่สุด 2 จุดสำหรับ interpolation
        if adc_value <= 235:
            # Extrapolate จากจุดแรกสองจุด
            x1, y1 = 235, 150
            x2, y2 = 364, 100
        elif adc_value <= 364:
            # Interpolate ระหว่างจุดแรกและจุดที่สอง
            x1, y1 = 235, 150
            x2, y2 = 364, 100
        elif adc_value <= 705:
            # Interpolate ระหว่างจุดที่สองและจุดที่สาม
            x1, y1 = 364, 100
            x2, y2 = 705, 50
        else:
            # Extrapolate จากจุดสุดท้ายสองจุด
            x1, y1 = 364, 100
            x2, y2 = 705, 50
        
        # Linear interpolation: y = y1 + (y2-y1) * (x-x1) / (x2-x1)
        distance = y1 + (y2 - y1) * (adc_value - x1) / (x2 - x1)
        
        return max(20, min(300, distance))
    # ========================================================================
    # Sensor Scanning Functions
    # ========================================================================s

    def _get_neighbor_position(self, scan_direction):
        """คำนวณตำแหน่งเพื่อนบ้านตามทิศทาง"""
        x, y = self.current_position
        if scan_direction == 0: return (x, y + 1)      # North
        elif scan_direction == 1: return (x + 1, y)    # East  
        elif scan_direction == 2: return (x, y - 1)    # South
        elif scan_direction == 3: return (x - 1, y)    # West
        return None
    
    def _detect_wall_in_direction(self, scan_direction):
        """ตรวจจับกำแพงในทิศทางที่กำหนด"""
        relative_direction = (scan_direction - self.current_orientation + 4) % 4
        
        if relative_direction == 0:  # ด้านหน้า -> ใช้ ToF
            self.ep_gimbal.moveto(yaw=0, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
            time.sleep(1)
            distance_mm = self.tof_handler.get_distance()
            has_wall = distance_mm < WALL_THRESHOLD_MM
            print(f"         - ToF Front: {distance_mm:.1f} mm, Wall: {has_wall}")
            
            return has_wall, distance_mm
            
        elif relative_direction == 1:  # ด้านขวา -> ใช้ IR ขวา
            self.ep_gimbal.moveto(yaw=90, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
            time.sleep(1)
            distance_mm = self.tof_handler.get_distance()
            has_wall = distance_mm < WALL_THRESHOLD_MM
            print(f"         - ToF Front: {distance_mm:.1f} mm, Wall: {has_wall}")
            
            return has_wall, distance_mm
            
        elif relative_direction == 3:  # ด้านซ้าย -> ใช้ IR ซ้าย
            self.ep_gimbal.moveto(yaw=-90, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
            time.sleep(1)
            distance_mm = self.tof_handler.get_distance()
            has_wall = distance_mm < WALL_THRESHOLD_MM
            print(f"         - ToF Front: {distance_mm:.1f} mm, Wall: {has_wall}")
            
            return has_wall, distance_mm
            
        elif relative_direction == 2 and self.count == True:
            self.ep_gimbal.moveto(yaw=180, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
            time.sleep(1)
            distance_mm = self.tof_handler.get_distance()
            has_wall = distance_mm < WALL_THRESHOLD_MM
            print(f"         - ToF Front: {distance_mm:.1f} mm, Wall: {has_wall}")
            self.count = False
            return has_wall, distance_mm
    
    def scan_surroundings_with_gimbal(self, previous_position=None):
        """
        สแกนหากำแพงรอบๆ โดยใช้ ToF (หน้า) และ IR sensors (ซ้าย-ขวา)
        
        Args:
            previous_position: ตำแหน่งก่อนหน้า (ไม่ต้องสแกนทิศนั้น)
            
        Returns:
            dict: ระยะทางที่วัดได้ในแต่ละทิศ
        """
        self.internal_map.mark_explored(self.current_position)
        
        wall_distances = {}
        self.periodic_wall_clearance_adjust(target_clearance_m=0.16)
        direction_to_skip = -1
        if previous_position:
            print(f"   -> Path from {previous_position} is known. Adding connection automatically.")
            self.internal_map.add_connection(self.current_position, previous_position)
            direction_to_skip = (self.current_orientation + 2) % 4
            print(f"   -> Will skip physical scan for direction {ORIENTATIONS.get(direction_to_skip, 'N/A')}.")

        # อ่านข้อมูล IR sensors ครั้งเดียว

        # สแกนทั้ง 4 ทิศ
        for scan_direction in range(4):
            if scan_direction == direction_to_skip:
                continue
                
            print(f"   Scanning direction: {ORIENTATIONS[scan_direction]}...")
            
            neighbor_pos = self._get_neighbor_position(scan_direction)
            has_wall, distance_mm = self._detect_wall_in_direction(scan_direction)
            
            wall_distances[f'{scan_direction}'] = distance_mm

            # ประมวลผลผลการตรวจจับ
            if not has_wall:
                # ไม่มีกำแพง -> เป็นทางเปิด
                if (neighbor_pos[0] >= 0 and neighbor_pos[1] >= 0 and 
                    neighbor_pos[0] < self.border[0] and neighbor_pos[1] < self.border[1]):
                    self.internal_map.add_connection(self.current_position, neighbor_pos)
                    print(f"           - Open path recorded towards {neighbor_pos}.")
            else:
                # มีกำแพง -> บล็อกทาง
                print(f"           - Wall detected. Blocking path to {neighbor_pos}.")
                self.internal_map.add_blocked(self.current_position, neighbor_pos)
                self.wall_log.add(tuple(sorted([self.current_position, neighbor_pos])))

        # รีเซ็นเตอร์กิมบอล
        self.ep_gimbal.moveto(yaw=0, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
        print("Scan complete using ToF and IR sensors.")

        self._save_scan_results(wall_distances)



        return wall_distances
    
    def _save_scan_results(self, wall_distances):
        """บันทึกผลการสแกนลงใน scan_log"""
        x, y = self.current_position
        self.scan_log.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "grid_x": x, "grid_y": y,
            "N_mm": wall_distances.get('0', None),
            "E_mm": wall_distances.get('1', None),
            "S_mm": wall_distances.get('2', None),
            "W_mm": wall_distances.get('3', None)
        })
    # -------------------- NEW: Target & Map JSON helpers --------------------
    def record_target_hit(self, color, shape):
        """บันทึกเหตุการณ์พบ/ยิงเป้าหมาย ณ กริดปัจจุบัน"""
        gx, gy = self.current_position
        x_m = gx * GRID_SIZE_M
        y_m = gy * GRID_SIZE_M
        _, _, _, yaw_deg, _, _ = self.pose_handler.get_pose()
        # Check for duplicate based on color, shape, grid, and orientation
        duplicate = any(
            t.get("color") == color and
            t.get("shape") == shape and
            t.get("grid", {}).get("x") == gx and
            t.get("grid", {}).get("y") == gy and
            t.get("orientation") == int(self.current_orientation)
            for t in self.detected_targets
        )
        if not duplicate:
            self.detected_targets.append({
                "color": color,
                "shape": shape,
                "grid": {"x": gx, "y": gy},
                "orientation": int(self.current_orientation),     # 0=N,1=E,2=S,3=W
                "orientation_name": ORIENTATIONS.get(self.current_orientation, "?")
            })

    def save_targets_json(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.targets_json_path = f"targets_run1_{ts}.json"
        payload = {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "grid_size_m": GRID_SIZE_M
            },
            "targets": self.detected_targets
        }
        try:
            with open(self.targets_json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[JSON] Targets saved → {self.targets_json_path}")
        except Exception as e:
                print(f"{e}")

    def save_map_state_json(self):
        """เซฟกราฟ/กำแพง/พื้นที่ที่สำรวจ/เส้นทาง สำหรับ Lap2"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.map_state_json_path = f"map_state_run1_{ts}.json"

        # แปลง graph (set/tuple) ให้อยู่ในรูปที่ serialize ได้
        graph_json = {f"{k[0]},{k[1]}": [[n[0], n[1]] for n in v]
                      for k, v in self.internal_map.graph.items()}
        blocked_json = [[[a[0], a[1]], [b[0], b[1]]]
                        for (a, b) in sorted(self.internal_map.blocked)]
        explored_json = [[p[0], p[1]] for p in sorted(self.internal_map.explored)]
        visited_json = [[p[0], p[1]] for p in self.visited_path]

        gx, gy = self.current_position
        final_world = {"x": gx * GRID_SIZE_M, "y": gy * GRID_SIZE_M}

        payload = {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "grid_size_m": GRID_SIZE_M,
                "border": {"w": self.border[0], "h": self.border[1]},
                # เปลี่ยนจาก final_yaw → final_orientation (ตามที่เคยบอก)
                "final_orientation": int(self.current_orientation),
                "final_orientation_name": ORIENTATIONS.get(self.current_orientation, "?"),
                "final_position_grid": {"x": gx, "y": gy},
            },
            "graph": graph_json,
            "blocked": blocked_json,
            "explored": explored_json,
            "visited_path": visited_json
        }

        try:
            with open(self.map_state_json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[JSON] Map state saved → {self.map_state_json_path}")
        except Exception as e:
            print(f"[JSON] Error saving map state: {e}")

    # ========================================================================
    # Navigation Functions
    # ========================================================================
    

    def decide_next_path(self):
        """
        ตัดสินใจหาเส้นทางถัดไป - สำรวจทั้งหมดเท่านั้น
        
        Returns:
            list: เส้นทางที่ต้องเดิน หรือ None หากสำรวจครบแล้ว
        """
        # หาจุดใหม่ที่ยังไม่ได้สำรวจ
        unexplored = self.internal_map.get_unexplored_neighbors(self.current_position)
        if unexplored:
            return [self.current_position, unexplored[0]]
        
        # ไม่มีจุดใหม่รอบๆ - ย้อนกลับหาจุดที่มีทางใหม่
        for pos in reversed(self.visited_path):
            if self.internal_map.get_unexplored_neighbors(pos):
                print(f"Backtracking to find unexplored path from {pos}...")
                return self.internal_map.get_path(self.current_position, pos)
        
        # ไม่มีจุดใหม่แล้ว - สำรวจครบแล้ว
        print(f"\n=== EXPLORATION COMPLETE! ===")
        print(f"All reachable areas have been explored.")
        self.exploration_complete = True
        return None

    # ========================================================================
    # Real-time Plotting Functions
    # ========================================================================
    
    def init_real_time_plot(self):
        """เริ่มต้น real-time plotting"""
        if not self.real_time_plot:
            return
            
        try:
            plt.ion()  # เปิด interactive mode
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_title("Robot Map - Real Time")
            self.ax.set_aspect('equal', adjustable='box')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlim(-1, self.border[0])
            self.ax.set_ylim(-1, self.border[1])
            plt.show(block=False)
            print("Real-time plotting initialized")
        except Exception as e:
            print(f"Failed to initialize real-time plot: {e}")
            self.real_time_plot = False

    def update_real_time_plot(self):
        """อัปเดต real-time plot"""
       # ปิดทันทีถ้าภารกิจสำเร็จ
        if getattr(self, "exploration_complete", False):
           self.close_real_time_plot()
           return
        if not self.real_time_plot or self.fig is None:
            return
            
        try:
            # ล้างแผนที่เก่า
            self.ax.clear()
            self.ax.set_title(f"Robot Map - Real Time | Position: {self.current_position}")
            self.ax.set_aspect('equal', adjustable='box')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlim(-1, self.border[0])
            self.ax.set_ylim(-1, self.border[1])

            # วาด "ทางเปิด" จาก graph (เส้นบาง)
            drawn = set()
            for a, nbs in self.internal_map.graph.items():
                for b in nbs:
                    edge = tuple(sorted([a, b]))
                    if edge in drawn:
                        continue
                    drawn.add(edge)
                    x1, y1 = a
                    x2, y2 = b
                    self.ax.plot([x1, x2], [y1, y2], 'b-', linewidth=1.2, alpha=0.5, zorder=1)

            # วาด "กำแพงปิด" จาก blocked (เส้นหนา)
            for edge in self.internal_map.blocked:
                (x1, y1), (x2, y2) = edge
                dx, dy = x2 - x1, y2 - y1
                if dx == 0 and abs(dy) == 1:
                    # แนวเดียวกันคอลัมน์ (เหนือ-ใต้) -> กำแพงแนวนอนที่กึ่งกลาง
                    y_mid = (y1 + y2) / 2.0
                    self.ax.plot([x1 - 0.5, x1 + 0.5], [y_mid, y_mid], 'k-', linewidth=3.0, zorder=3)
                elif dy == 0 and abs(dx) == 1:
                    # แนวเดียวกันแถว (ซ้าย-ขวา) -> กำแพงแนวตั้งที่กึ่งกลาง
                    x_mid = (x1 + x2) / 2.0
                    self.ax.plot([x_mid, x_mid], [y1 - 0.5, y1 + 0.5], 'k-', linewidth=3.0, zorder=3)

            # วาดโหนดที่สำรวจแล้ว
            if self.internal_map.graph:
                explored_nodes = list(self.internal_map.explored)
                if explored_nodes:
                    ex, ey = zip(*explored_nodes)
                    self.ax.scatter(ex, ey, s=50, c='lightblue', marker='s', zorder=2, label='Explored')

            # วาดเส้นทางที่เดิน
            if len(self.visited_path) > 1:
                px, py = zip(*self.visited_path)
                self.ax.plot(px, py, 'r-', linewidth=2, zorder=4, label='Path')

            # วาดตำแหน่งปัจจุบัน
            cx, cy = self.current_position
            self.ax.scatter([cx], [cy], s=150, c='red', marker='o', zorder=5, label='Current')

            # วาดจุดเริ่มต้น
            if self.visited_path:
                sx, sy = self.visited_path[0]
                self.ax.scatter([sx], [sy], s=120, c='green', marker='o', zorder=5, label='Start')

            # วาด object markers แบบเดียวกับ plot_map_with_walls
            if self.shoot_grid:
                blocked_edges = set(self.internal_map.blocked)
                width, height = self.border

                def border_center(x, y, orientation):
                    if orientation == 0:   return (x, height - 0.5)   # ชนขอบบน
                    if orientation == 2:   return (x, -0.5)           # ชนขอบล่าง
                    if orientation == 1:   return (width - 0.5, y)    # ชนขอบขวา
                    if orientation == 3:   return (-0.5, y)           # ชนขอบซ้าย
                    return (x, y)

                for entry in self.shoot_grid:
                    # รองรับทั้งโครงสร้างเก่าและใหม่
                    if len(entry) == 4 and isinstance(entry[2], str):
                        pos, orientation, color, shape = entry
                        cs_iter = [(color, shape)]
                    elif len(entry) == 3 and isinstance(entry[2], list):
                        pos, orientation, cs_iter = entry
                    else:
                        continue

                    x, y = pos
                    if orientation == 0:
                        dx, dy = 0, 1
                    elif orientation == 1:
                        dx, dy = 1, 0
                    elif orientation == 2:
                        dx, dy = 0, -1
                    else:
                        dx, dy = -1, 0

                    cx, cy = x, y
                    mx = my = None
                    while True:
                        nx, ny = cx + dx, cy + dy
                        out_of_bounds = not (0 <= nx < width and 0 <= ny < height)
                        if out_of_bounds:
                            mx, my = border_center(cx, cy, orientation)
                            break
                        edge = tuple(sorted([(cx, cy), (nx, ny)]))
                        if edge in blocked_edges:
                            mx, my = (cx + nx) / 2.0, (cy + ny) / 2.0
                            break
                        cx, cy = nx, ny

                    if mx is None or my is None:
                        continue

                    # นำ offset ตามทิศ เพื่อไม่ให้ดาวทับเส้นกำแพง
                    offset = 0.10
                    if orientation == 0:
                        ox, oy = 0.0, -offset
                    elif orientation == 1:
                        ox, oy = -offset, 0.0
                    elif orientation == 2:
                        ox, oy = 0.0, offset
                    else:
                        ox, oy = offset, 0.0
                    mx_off, my_off = mx + ox, my + oy

                    for color, shape in cs_iter:
                        self.ax.scatter(mx_off, my_off, color='gold', s=180, marker='*', edgecolors='black', zorder=6)
                        self.ax.text(mx_off, my_off - 0.06, f"{color} {shape}", color='black', ha='center', va='top', fontsize=9, zorder=7)

            self.ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
            
            # อัปเดตแผนที่
            plt.draw()
            plt.pause(0.1)  # หน่วงเวลาเล็กน้อยให้เห็นการเปลี่ยนแปลง
            
        except Exception as e:
            print(f"Error updating real-time plot: {e}")

    def close_real_time_plot(self):
        try:
            if self.fig is not None:
                plt.ioff()
                plt.close(self.fig)
            # กันถูกเรียกอัปเดตซ้ำหลังปิดแล้ว
            self.fig = None
            self.ax = None
            self.real_time_plot = False
            print("Real-time plotting closed")
        except Exception as e:
            print(f"Error closing real-time plot: {e}")

    # ========================================================================
    # Sensor Adjustment Functions
    # ========================================================================
    
    def periodic_wall_clearance_adjust(self, target_clearance_m=0.175):
        """
        ปรับตำแหน่งหุ่นโดยใช้ข้อมูลจากเซ็นเซอร์ ToF
        เพื่อให้อยู่ห่างจากกำแพงในระยะที่เหมาะสม
        """
        
        self.ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()

        # ToF หน้า
        
        time.sleep(0.5)
        # ปรับตำแหน่งถ้าใกล้กำแพงหน้าเกินไป
        self.ep_gimbal.moveto(yaw=0, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
        time.sleep(0.5)
        front_distance_mm = self.tof_handler.get_distance() 
        if front_distance_mm < WALL_THRESHOLD_MM:
            move_dist_m = (front_distance_mm / 1000.0) - target_clearance_m
            if abs(move_dist_m) > 0.05:
                self.ep_chassis.move(x=move_dist_m, y=0, z=0, xy_speed=0.5).wait_for_completed()
            return 
        self.ep_gimbal.moveto(yaw=-90, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
        time.sleep(0.5)
        left_distance_mm = self.tof_handler.get_distance() 
        if left_distance_mm < WALL_THRESHOLD_MM:
            move_dist_m = (left_distance_mm / 1000.0) - target_clearance_m-0.02
            print(f"left {-move_dist_m}")
            if abs(move_dist_m) > 0.05:
                self.ep_chassis.move(x=0, y=-move_dist_m, z=0, xy_speed=0.5).wait_for_completed()
        self.ep_gimbal.moveto(yaw=90, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
        time.sleep(0.5)
        right_distance_mm = self.tof_handler.get_distance()
        if right_distance_mm < WALL_THRESHOLD_MM:
            move_dist_m = (right_distance_mm / 1000.0) - target_clearance_m-0.02
            print(f"right {move_dist_m}")
            if abs(move_dist_m) > 0.05:
                self.ep_chassis.move(x=0, y=move_dist_m, z=0, xy_speed=0.5).wait_for_completed()
        self.ep_chassis.move(x=0, y=0, z=0, xy_speed=0.5).wait_for_completed()

    # ========================================================================
    # Movement and PID Control Functions
    # ========================================================================
    
    def move_forward_pid(self, distance_m, speed_limit=0.6):
        """
        ใช้ PID controller เพื่อเคลื่อนที่ไปข้างหน้าระยะทางที่กำหนด
        
        Args:
            distance_m: ระยะทางที่ต้องการเดิน (เมตร)
            speed_limit: ความเร็วสูงสุด (เมตร/วินาที)
        """
        print(f"   PID Move: Moving forward {distance_m}m.")
        pid = PIDController(Kp=2, Ki=0.02, Kd=0.1, setpoint=distance_m, output_limits=(-speed_limit, speed_limit))
        start_x, start_y, _, _, _, _ = self.pose_handler.get_pose()
        # Execute motion in small, blocking steps to avoid overlapping actions
        max_step = 0.6  # meters per step
        while True:
            # Obstacle check BEFORE issuing a new move action
            if self.tof_handler.get_distance() <= 280:
                
                print("   PID Move: Obstacle detected by ToF (<=280mm). Aborting forward move.")
                break

            curr_x, curr_y, _, _, _, _ = self.pose_handler.get_pose()
            dist_traveled = math.hypot(curr_x - start_x, curr_y - start_y)
            print(dist_traveled)
            remaining = distance_m - dist_traveled
            if abs(remaining) < 0.1:
                break

            # Determine a safe step distance and speed
            step = max(-max_step, min(max_step, remaining))
            step_speed = min(speed_limit, max(0.6, abs(step) * 2))

            action = self.ep_chassis.move(x=step, y=0, z=0, xy_speed=step_speed)
            action.wait_for_completed()
            self.ep_chassis.move(x=0, y=0, z=0, xy_speed=step_speed).wait_for_completed
            time.sleep(0.01)
        print("   PID Move: Completed.")

        gx, gy = self.current_position
        _, _, _, yaw, _, _ = self.pose_handler.get_pose()
        self.path_log.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "grid_x": gx, "grid_y": gy, "yaw": yaw
        })

    def turn_pid(self, target_angle, speed_limit=90):
        """
        ใช้ PID controller เพื่อหมุนไปยังมุมที่กำหนด
        
        Args:
            target_angle: มุมเป้าหมาย (องศา)
            speed_limit: ความเร็วการหมุนสูงสุด (องศา/วินาที)
        """
        print(f"   PID Turn: Turning to {target_angle} degrees.")
        # Execute rotation in small, blocking steps to avoid overlapping actions
        max_step_deg = 30  # degrees per step
        while True:
            _, _, _, current_yaw, _, _ = self.pose_handler.get_pose()
            error = target_angle - current_yaw
            if error > 180: error -= 360
            if error < -180: error += 360
            print(error,2)
            if abs(error) < 1.5:
                break
            
            # Use a clamped step rotation to the target
            step_deg = max(-max_step_deg, min(max_step_deg, error))
            step_speed = min(speed_limit, max(30, abs(step_deg) * 10))

            action = self.ep_chassis.move(x=0, y=0, z=-step_deg, z_speed=step_speed)
            action.wait_for_completed()
            time.sleep(0.01)
        self.ep_gimbal.moveto(yaw=0, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
        time.sleep(0.5)
        print("   PID Turn: Completed.")

    def execute_path(self, path):
        """
        ดำเนินการเดินตามเส้นทางที่กำหนดด้วย PID control
        
        Args:
            path: รายการของตำแหน่งที่ต้องเดิน [(x1,y1), (x2,y2), ...]
        """
        if not path or len(path) < 2: return
        print(f"Executing path with PID: {path}")
        for i in range(len(path) - 1):
            self.periodic_wall_clearance_adjust(target_clearance_m=0.16)
            self.visited_for_check.append(self.current_position)
            start_node, end_node = path[i], path[i+1]
            dx, dy = end_node[0] - start_node[0], end_node[1] - start_node[1]
            target_orientation = -1
            if dx == 0 and dy == 1: target_orientation = 0
            elif dx == 1 and dy == 0: target_orientation = 1
            elif dx == 0 and dy == -1: target_orientation = 2
            elif dx == -1 and dy == 0: target_orientation = 3

            target_angle = 0
            if target_orientation == 1: target_angle = 90
            elif target_orientation == 2: target_angle = 180
            elif target_orientation == 3: target_angle = -90
            self.detect_and_shot()
            before_orientation = self.current_orientation
            self.turn_pid(target_angle)
            self.current_orientation = target_orientation
            if before_orientation != self.current_orientation:
                self.periodic_wall_clearance_adjust(target_clearance_m=0.16)
                self.detect_and_shot()
            
            time.sleep(0.2)
            self.move_forward_pid(GRID_SIZE_M)
            self.current_position = end_node
            self.visited_path.append(self.current_position)
    
            self.pose_handler.set_xy(end_node[0] * GRID_SIZE_M, end_node[1] * GRID_SIZE_M)
            self.pose_handler.set_yaw(target_angle)
            time.sleep(0.2)
            print("end++++++++++++++++++++++++++++++++++++++++++++++++++")

    # ========================================================================
    # Main Mission Control
    # ========================================================================
    
    def run_mission(self):
        """
        ดำเนินการภารกิจหลัก:
        1. สำรวจพื้นที่ทั้งหมด
        2. แสดงแผนที่แบบ real-time
        3. บันทึกข้อมูลเซ็นเซอร์และเส้นทาง
        """
        try:
            self.ep_gimbal.moveto(yaw=0, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
            
            # เริ่มต้น real-time plotting
            self.init_real_time_plot()
            self.update_real_time_plot()  # แสดงแผนที่เริ่มต้น
            """
            รันการตรวจจับวัตถุด้วยกล้อง RoboMaster แบบสั้นๆ
            
            target_color: สีเป้าหมาย ('red', 'green', 'blue', 'yellow')
            target_shape: รูปทรงเป้าหมาย
                - 'circle': วงกลม
                - 'square': สี่เหลี่ยมจัตุรัส  
                - 'vertical_rectangle': สี่เหลี่ยมผืนผ้าแนวตั้ง
                - 'horizontal_rectangle': สี่เหลี่ยมผืนผ้าแนวนอน
            """
            self.ep_camera.start_video_stream(display=False, resolution="720p")

            self.set_target('blue', 'square')
            while True:
                
                if self.current_position not in self.internal_map.explored:
                    previous_pos = self.visited_path[-2] if len(self.visited_path) > 1 else None
                    self.scan_surroundings_with_gimbal(previous_position=previous_pos)
                    # อัปเดตแผนที่หลังสแกน
                    self.update_real_time_plot()

                path_to_execute = self.decide_next_path()
                if not path_to_execute:
                    print(f"\n--- EXPLORATION COMPLETE! All areas explored ---")
                    
                    
                    self.close_real_time_plot()
                    break
                self.execute_path(path_to_execute)
                # อัปเดตแผนที่หลังเคลื่อนที่
                self.update_real_time_plot()
        except Exception as e:
                print(f"{e}")
        finally:
            self.save_targets_json()
            self.save_map_state_json()
            print(self.shoot_grid)
            print("\n--- Final Marker Map ---")

            # ปิด real-time plot และบันทึกแผนที่สุดท้าย
            self.close_real_time_plot()
            
            self.plot_map_with_walls(filename="maze_map.png")

            self.save_csv_logs()
            self.ep_camera.stop_video_stream()

# ==============================================================================
# Map Visualization Functions
# ==============================================================================

    def plot_map_with_walls(self, filename="maze_map.png"):
        """วาดแผนที่สุดท้ายพร้อมกำแพง เส้นทาง และ markers โดยใช้สถานะภายในออบเจ็กต์"""
        graph = self.internal_map.graph
        blocked = self.internal_map.blocked
        path = self.visited_path
        width, height = self.border

        plt.figure(figsize=(8, 8))
        plt.title("Robot Map with Walls and Path")

        # วาดเส้นทางเปิดจากกราฟ (หลีกเลี่ยงซ้ำขอบเดียวกัน)
        drawn = set()
        for a, nbs in graph.items():
            for b in nbs:
                edge = tuple(sorted([a, b]))
                if edge in drawn:
                    continue
                drawn.add(edge)
                x1, y1 = a
                x2, y2 = b
                plt.plot([x1, x2], [y1, y2], linewidth=1.2, alpha=0.5, zorder=1)

        # วาดกำแพงจาก blocked edges
        for (a, b) in sorted(blocked):
            (x1, y1), (x2, y2) = a, b
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and abs(dy) == 1:
                y_mid = (y1 + y2) / 2.0
                plt.plot([x1 - 0.5, x1 + 0.5], [y_mid, y_mid], linewidth=3.0, color="k", zorder=3)
            elif dy == 0 and abs(dx) == 1:
                x_mid = (x1 + x2) / 2.0
                plt.plot([x_mid, x_mid], [y1 - 0.5, y1 + 0.5], linewidth=3.0, color="k", zorder=3)

        # วาดโหนด
        if graph:
            xs, ys = zip(*graph.keys())
            plt.scatter(xs, ys, s=36, color="tab:blue", zorder=2, label="Nodes")

        # วาดเส้นทางที่เดิน
        if path:
            px, py = zip(*path)
            plt.plot(px, py, linewidth=2, color="tab:red", zorder=4, label="Path")
            plt.scatter(px[0], py[0], s=120, color="green", marker='o', zorder=5, label='Start')
            plt.scatter(px[-1], py[-1], s=120, color="purple", marker='X', zorder=5, label='End')

        # วาด markers จาก shoot_grid (รองรับทั้งโครงสร้างเก่าและใหม่)
        if self.shoot_grid:
            blocked_edges = set(blocked)
            markers_by_wall = {}

            def add_marker(mx, my, text):
                key = (mx, my)
                markers_by_wall.setdefault(key, []).append(text)

            def border_center(x, y, orientation):
                if orientation == 0:   return (x, height - 0.5)   # North border
                if orientation == 2:   return (x, -0.5)           # South border
                if orientation == 1:   return (width - 0.5, y)    # East border
                if orientation == 3:   return (-0.5, y)           # West border
                return (x, y)

            for entry in self.shoot_grid:
                # โครงสร้างเก่า: (pos, orientation, color, shape)
                if len(entry) == 4 and isinstance(entry[2], str):
                    pos, orientation, color, shape = entry
                    cs_iter = [(color, shape)]
                # โครงสร้างใหม่: (pos, orientation, [(color,shape), ...])
                elif len(entry) == 3 and isinstance(entry[2], list):
                    pos, orientation, cs_iter = entry
                else:
                    continue

                x, y = pos
                # ทิศ -> เวกเตอร์การเดิน
                if orientation == 0:
                    dx, dy = 0, 1
                elif orientation == 1:
                    dx, dy = 1, 0
                elif orientation == 2:
                    dx, dy = 0, -1
                else:
                    dx, dy = -1, 0

                cx, cy = x, y
                mx = my = None
                while True:
                    nx, ny = cx + dx, cy + dy
                    out_of_bounds = not (0 <= nx < width and 0 <= ny < height)
                    if out_of_bounds:
                        mx, my = border_center(cx, cy, orientation)
                        break
                    edge = tuple(sorted([(cx, cy), (nx, ny)]))
                    if edge in blocked_edges:
                        mx, my = (cx + nx) / 2.0, (cy + ny) / 2.0
                        break
                    cx, cy = nx, ny

                if mx is None or my is None:
                    continue

                # Apply a small offset to move the star away from the wall/border for visibility
                offset = 0.10
                if orientation == 0:      # N → nudge slightly downward (into the cell)
                    ox, oy = 0.0, -offset
                elif orientation == 1:    # E → nudge slightly left
                    ox, oy = -offset, 0.0
                elif orientation == 2:    # S → nudge slightly upward
                    ox, oy = 0.0, offset
                else:                      # W → nudge slightly right
                    ox, oy = offset, 0.0

                mx_off, my_off = mx + ox, my + oy
                for color, shape in cs_iter:
                    add_marker(mx_off, my_off, f"{color} {shape}")

            # วาด marker รวมบนผนังตำแหน่งเดียวกัน
            for (mx, my), texts in markers_by_wall.items():
                plt.scatter(mx, my, color='gold', s=180, marker='*', edgecolors='black', zorder=6)
                # Place text just below the star marker for readability
                plt.text(mx, my - 0.06, '\n'.join(texts), color='black', ha='center', va='top', fontsize=9, zorder=7)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.savefig(filename, dpi=150)
        print(f"Map saved to '{filename}'")
        plt.close()




if __name__ == '__main__':

    # เชื่อมต่อหุ่นยนต์
    print("Connecting to RoboMaster robot...")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    print("Robot connected successfully.")

    # สร้าง data handlers สำหรับเซ็นเซอร์ต่างๆ
    tof_handler = TofDataHandler()
    gimbal_handler = GimbalDataHandler()
    pose_handler = PoseDataHandler()

    # สมัครรับข้อมูลจากเซ็นเซอร์
    print("Setting up sensor subscriptions...")
    ep_robot.sensor.sub_distance(freq=10, callback=tof_handler.update)
    ep_robot.chassis.sub_position(freq=10, callback=pose_handler.update_position)
    ep_robot.chassis.sub_attitude(freq=10, callback=pose_handler.update_attitude)
    ep_robot.gimbal.sub_angle(freq=20, callback=gimbal_handler.update)
    print("All sensor subscriptions completed.")


    explorer = MazeExplorer(ep_robot, tof_handler, gimbal_handler, pose_handler)

    explorer.start_continuous_logging(frequency_hz=10)

    # รอให้ระบบเซ็นเซอร์เสถียร
    time.sleep(2)
    try:
        explorer.run_mission()
    except Exception as e:
                print(f"{e}")
    finally:
        try:
            explorer.save_targets_json()
            explorer.save_map_state_json()
        except Exception as e:
                print(f"{e}")
    # การหยุดทำงานอย่างปลอดภัย

    
    if explorer:
        explorer.stop_continuous_logging()
        explorer.save_csv_logs()
        explorer.close_real_time_plot()

    if ep_robot:

        try:
            ep_robot.sensor.unsub_distance()
            ep_robot.gimbal.unsub_angle()
            ep_robot.chassis.unsub_position()
            ep_robot.chassis.unsub_attitude()
            ep_robot.close()
            print("Robot connection closed safely.")
        except Exception as e:
            print(f"Warning during robot cleanup: {e}")
    
    print("✅ Mission completed and cleanup finished.")