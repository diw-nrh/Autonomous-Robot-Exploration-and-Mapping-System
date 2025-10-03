#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
project_3_common.py - Common functions and classes for Robot Maze Explorer

Contains all shared functionality between lab1 and lab2:
- Data handlers (ToF, Vision, Gimbal, Pose)
- Map and navigation classes
- PID controller
- Sensor functions
- Shooting functions  
- Target storage
- Plotting functions
"""

import robomaster
from robomaster import robot
import time
import math
import json
import csv
import matplotlib.pyplot as plt
import statistics
import threading
from collections import deque
from datetime import datetime
import sys

# Import compact vision detector
try:
    from compact_vision import CompactVisionDetector
except ImportError:
    print("Warning: compact_vision not available. Using fallback.")
    class CompactVisionDetector:
        def __init__(self): pass
        def set_target(self, color, shape): pass
        def detect(self, frame): return []

# =============================================================================
# HELPER FUNCTIONS FOR COMPACT VISION COMPATIBILITY
# =============================================================================

def quick_target_check(camera, color='red', shape='circle'):
    """ตรวจสอบเป้าหมายแบบเร็ว"""
    detector = CompactVisionDetector()
    detector.set_target(color, shape)
    
    try:
        frame = camera.read_cv2_image(strategy="newest", timeout=1.0)
        if frame is not None:
            detections = detector.detect(frame)
            return len(detections) > 0
    except:
        pass
    
    return False

def scan_for_targets(camera, color='red', shape='circle', duration=2.0):
    """สแกนหาเป้าหมายด้วยกล้อง"""
    detector = CompactVisionDetector()
    detector.set_target(color, shape)
    
    import time
    start_time = time.time()
    all_detections = []
    
    while time.time() - start_time < duration:
        try:
            frame = camera.read_cv2_image(strategy="newest", timeout=0.5)
            if frame is not None:
                detections = detector.detect(frame)
                all_detections.extend(detections)
        except:
            pass
        time.sleep(0.1)
    
    return all_detections

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

START_POSITION = (0, 0)   # จุดเริ่ม (grid_x, grid_y)

# Robot Movement and Grid Settings
GRID_SIZE_M = 0.635                    # ขนาดกริดเป็นเมตร
WALL_THRESHOLD_MM = 450                # เกณฑ์ระยะทางที่ถือว่ามีกำแพง (มม.)

BORDER = (4, 5)                        # ขนาดพื้นที่สำรวจ (กริด: กว้าง x สูง)
# Gimbal and Vision Settings
GIMBAL_TURN_SPEED = 360                # ความเร็วหมุนกิมบอล (องศา/วินาที)
CAMERA_HORIZONTAL_FOV = 96             # มุมมองกล้องแนวนอน (องศา)
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
# TARGET DATA STORAGE AND SHOOTING FUNCTIONS
# =============================================================================

class TargetStorage:
    """จัดการข้อมูลเป้าหมายที่พบในรอบ 1 เพื่อใช้ในรอบ 2"""
    
    def __init__(self):
        self.detected_targets = []
        self.storage_file = "detected_targets.json"
    
    def add_target(self, position, note="object", detection_method="vision", confidence=1.0):
        """เพิ่มเป้าหมายใหม่ที่ตรวจพบ"""
        target_data = {
            "position": list(position),
            "note": note,
            "detection_method": detection_method,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "shot": False  # ยังไม่ได้ยิง
        }
        
        # ตรวจสอบว่ามีเป้าที่ตำแหน่งนี้แล้วหรือไม่
        for existing in self.detected_targets:
            if tuple(existing["position"]) == tuple(position):
                existing["confidence"] = max(existing["confidence"], confidence)
                existing["note"] = note  # อัปเดตหมายเหตุ
                print(f"[TARGET STORAGE] อัปเดตเป้าที่ตำแหน่ง {position}: {note}")
                return
        
        self.detected_targets.append(target_data)
        print(f"[TARGET STORAGE] เพิ่มเป้าใหม่ที่ตำแหน่ง {position}: {note}")
        self.save_to_file()
    
    def get_all_targets(self):
        """คืนค่ารายการเป้าทั้งหมด"""
        return self.detected_targets
    
    def get_unshot_targets(self):
        """คืนค่าเป้าที่ยังไม่ได้ยิง"""
        return [t for t in self.detected_targets if not t["shot"]]
    
    def mark_target_shot(self, position):
        """ทำเครื่องหมายเป้าที่ยิงแล้ว"""
        for target in self.detected_targets:
            if tuple(target["position"]) == tuple(position):
                target["shot"] = True
                target["shot_timestamp"] = datetime.now().isoformat()
                print(f"[TARGET STORAGE] ทำเครื่องหมายเป้าที่ {position} ยิงแล้ว")
                self.save_to_file()
                return True
        return False
    
    def save_to_file(self):
        """บันทึกข้อมูลเป้าลงไฟล์"""
        try:
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(self.detected_targets, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[TARGET STORAGE] ข้อผิดพลาดในการบันทึก: {e}")
    
    def load_from_file(self):
        """โหลดข้อมูลเป้าจากไฟล์"""
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                self.detected_targets = json.load(f)
            print(f"[TARGET STORAGE] โหลดข้อมูล {len(self.detected_targets)} เป้าจากไฟล์")
            return True
        except FileNotFoundError:
            print("[TARGET STORAGE] ไม่พบไฟล์ข้อมูลเป้า")
            self.detected_targets = []
            return False
        except Exception as e:
            print(f"[TARGET STORAGE] ข้อผิดพลาดในการโหลด: {e}")
            self.detected_targets = []
            return False

# สร้าง global target storage instance
target_storage = TargetStorage()

def calculate_target_grid_position(robot_position, robot_orientation, detection_data):
    """
    คำนวณตำแหน่งกริดของวัตถุจากข้อมูลการตรวจจับ
    
    Args:
        robot_position: ตำแหน่งหุ่นยนต์ปัจจุบัน (grid_x, grid_y)
        robot_orientation: ทิศทางหุ่นยนต์ (0:N, 1:E, 2:S, 3:W)
        detection_data: ข้อมูลการตรวจจับจาก vision (dict)
                       ต้องมี: 'distance_m', 'angle_deg', 'confidence'
    
    Returns:
        tuple: ตำแหน่งกริดของวัตถุ (target_grid_x, target_grid_y)
    """
    
    # ตอนนี้ใช้การประมาณแบบง่าย (สามารถปรับปรุงได้)
    if detection_data is None:
        # ถ้าไม่มีข้อมูล vision ให้ใช้ตำแหน่งหุ่นยนต์
        print("[GRID CALC] ไม่มีข้อมูล vision - ใช้ตำแหน่งหุ่นยนต์")
        return robot_position
    
    try:
        # ดึงข้อมูลจาก detection
        distance_m = detection_data.get('distance_m', 0.635)  # default 1 grid
        angle_deg = detection_data.get('angle_deg', 0)        # relative to robot
        confidence = detection_data.get('confidence', 0.5)
        
        # คำนวณทิศทางสัมบูรณ์
        absolute_angle = (robot_orientation * 90 + angle_deg) % 360
        
        # แปลงเป็นพิกัดโลก (วิธีแบบง่าย)
        dx_m = distance_m * math.sin(math.radians(absolute_angle))
        dy_m = distance_m * math.cos(math.radians(absolute_angle))
        
        # แปลงเป็นกริด
        robot_x, robot_y = robot_position
        target_grid_x = robot_x + round(dx_m / GRID_SIZE_M)
        target_grid_y = robot_y + round(dy_m / GRID_SIZE_M)
        
        print(f"[GRID CALC] วัตถุที่ระยะ {distance_m:.2f}m มุม {angle_deg:.1f}° -> กริด ({target_grid_x}, {target_grid_y})")
        
        return (target_grid_x, target_grid_y)
        
    except Exception as e:
        print(f"[GRID CALC] ข้อผิดพลาด: {e} - ใช้ตำแหน่งหุ่นยนต์")
        return robot_position

def x(position, note="object", detection_method="vision", confidence=1.0):
    """บันทึกตำแหน่งเป้าหมายที่ตรวจพบจริง"""
    target_storage.add_target(position, note, detection_method, confidence)

def shoot_target(position, robot_instance=None, detection_data=None):
    """
    ยิงเป้าหมาย ณ ตำแหน่งที่กำหนดด้วย PID control
    
    เงื่อนไขการยิง:
    1. ต้องหันหน้าไปข้างหน้า (orientation = 0)
    2. ระยะ ToF ต้องไม่เกิน 1.2 m (1200 mm)
    3. ต้องพบเป้าหมายด้วย vision system
    
    Args:
        position: ตำแหน่งเป้าในกริด
        robot_instance: instance ของหุ่นยนต์
        detection_data: ข้อมูลการตรวจจับเพื่อใช้ในการเล็ง
    """
    print(f"[SHOOTING] เริ่มการยิงเป้าที่ตำแหน่ง {position}")
    
    if robot_instance is None:
        print(f"[SHOOTING] ⚠️ ไม่มี robot instance - จำลองการยิง")
        target_storage.mark_target_shot(position)
        return
    
    try:
        # ปรับ gimbal ให้อยู่ตำแหน่งศูนย์กลาง
        robot_instance.ep_gimbal.moveto(yaw=0, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
        time.sleep(0.3)
        
        # สแกนหาเป้าแบบละเอียดด้วย vision
        print("[SHOOTING] 🔍 สแกนหาเป้าหมายแบบละเอียด...")
        target_found = False
        final_target_position = None
        
        try:
            # ใช้ compact vision เพื่อยืนยันเป้า
            detections = scan_for_targets(robot_instance.ep_camera, color='red', shape='circle', duration=2.0)
            if detections:
                target_found = True
                print(f"[SHOOTING] ✅ ยืนยันเป้าหมาย: {detections}")
                
                # เลือกเป้าที่ confidence สูงสุด
                best_detection = max(detections, key=lambda x: x.get('confidence', 0))
                final_target_position = best_detection
        except Exception as e:
            print(f"[SHOOTING] ⚠️ ไม่สามารถใช้ vision ได้: {e}")
        
        if target_found and final_target_position:
            # ตรวจสอบเงื่อนไขก่อนยิง
            print("[SHOOTING] 🔍 ตรวจสอบเงื่อนไขการยิง...")
            
            # 1. ตรวจสอบทิศทาง - ต้องหันไปข้างหน้า (orientation = 0)
            current_orientation = robot_instance.current_orientation if hasattr(robot_instance, 'current_orientation') else 0
            if current_orientation != 0:
                print(f"[SHOOTING] ❌ ทิศทางไม่ถูกต้อง - ปัจจุบัน: {current_orientation}, ต้องการ: 0 (หน้า)")
                target_storage.mark_target_shot(position)
                return
            
            # 2. ตรวจสอบระยะ ToF - ต้องไม่เกิน 1.2 m (1200 mm)
            if hasattr(robot_instance, 'tof_handler') and robot_instance.tof_handler:
                tof_distance = robot_instance.tof_handler.get_distance()
                max_distance_mm = 1200  # 1.2 m
                
                print(f"[SHOOTING] 📏 ระยะ ToF: {tof_distance:.1f} mm (สูงสุด: {max_distance_mm} mm)")
                
                if tof_distance > max_distance_mm:
                    print(f"[SHOOTING] ❌ ระยะไกลเกินไป - ToF: {tof_distance:.1f} mm > {max_distance_mm} mm")
                    target_storage.mark_target_shot(position)
                    return
                else:
                    print(f"[SHOOTING] ✅ ระยะเหมาะสม - ToF: {tof_distance:.1f} mm")
            else:
                print("[SHOOTING] ⚠️ ไม่มี ToF sensor - ข้ามการตรวจสอบระยะ")
            
            print("[SHOOTING] ✅ ผ่านเงื่อนไขทั้งหมด - เริ่มการเล็ง")
            
            # ใช้ PID เพื่อปรับทิศทางให้ตรงเป้า
            print("[SHOOTING] 🎯 ปรับทิศทางแม่นยำด้วย PID...")
            
            # สร้าง PID controller สำหรับการปรับทิศทาง (yaw และ pitch)
            yaw_pid = PIDController(Kp=3.0, Ki=0.1, Kd=0.2, setpoint=0.0, output_limits=(-45, 45))
            pitch_pid = PIDController(Kp=2.5, Ki=0.08, Kd=0.15, setpoint=0.0, output_limits=(-20, 20))
            
            # ปรับทิศทาง gimbal ให้ตรงเป้า
            target_x_offset = final_target_position.get('x_offset', 0)  # -1 ถึง 1 (ซ้าย-ขวา)
            target_y_offset = final_target_position.get('y_offset', 0)  # -1 ถึง 1 (บน-ล่าง)
            
            for attempt in range(15):  # ปรับสูงสุด 15 รอบ (เพิ่มจาก 10)
                try:
                    # ตรวจสอบว่าเล็งตรงเป้าแล้วหรือยัง (ทั้ง x และ y)
                    if abs(target_x_offset) < 0.05 and abs(target_y_offset) < 0.05:
                        print(f"[SHOOTING] ✅ เล็งตรงเป้าแล้ว (x_offset: {target_x_offset:.3f}, y_offset: {target_y_offset:.3f})")
                        break
                    
                    # ใช้ PID คำนวณการปรับทั้ง yaw และ pitch
                    yaw_adjustment = yaw_pid.update(target_x_offset)
                    pitch_adjustment = pitch_pid.update(-target_y_offset)  # ลบเพราะ pitch ทำงานตรงข้าม
                    
                    # ปรับ gimbal
                    current_yaw, current_pitch = robot_instance.gimbal_handler.get_yaw_pitch()
                    new_yaw = current_yaw + yaw_adjustment
                    new_pitch = current_pitch + pitch_adjustment
                    
                    # จำกัดช่วงการเคลื่อนไหว
                    new_yaw = max(-180, min(180, new_yaw))
                    new_pitch = max(-25, min(25, new_pitch))  # จำกัด pitch ให้อยู่ในช่วงปลอดภัย
                    
                    robot_instance.ep_gimbal.moveto(
                        yaw=new_yaw, 
                        pitch=new_pitch, 
                        yaw_speed=120, 
                        pitch_speed=100
                    ).wait_for_completed()
                    time.sleep(0.1)
                    
                    # สแกนใหม่เพื่อตรวจสอบตำแหน่ง
                    quick_scan = quick_target_check(robot_instance.ep_camera, color='red', shape='circle')
                    if quick_scan:
                        # อัปเดต offset (ต้องได้ข้อมูลจาก vision จริง)
                        target_x_offset *= 0.8  # ลดลงเรื่อยๆ สำหรับการจำลอง
                        target_y_offset *= 0.8
                    
                    print(f"[SHOOTING] 🎯 ปรับ gimbal: yaw={new_yaw:.1f}°, pitch={new_pitch:.1f}°, x_offset={target_x_offset:.3f}, y_offset={target_y_offset:.3f}")
                    
                except Exception as e:
                    print(f"[SHOOTING] ข้อผิดพลาดในการปรับทิศ: {e}")
                    break
            
            # ดำเนินการยิง
            print("[SHOOTING] 🔥 ยิง!")
            robot_instance.ep_led.set_led(r=255, g=0, b=0, effect="flash")
            
            # จำลองการยิง (แทนที่ด้วยคำสั่งยิงจริง)
            # TODO: เพิ่มคำสั่งยิงจริง เช่น robot_instance.ep_blaster.fire()
            time.sleep(0.5)  # เวลาการยิง
            
            # ใช้เสียงสัญญาณ (ถ้ามี)
            try:
                if hasattr(robot_instance, 'ep_speaker') and robot_instance.ep_speaker:
                    robot_instance.ep_speaker.play_sound(sound_id=1)  # เสียงยิง
            except:
                pass
            
            # รีเซ็ต gimbal กลับตำแหน่งเดิม
            robot_instance.ep_gimbal.moveto(yaw=0, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
            
            # เปลี่ยนสี LED เป็นเขียวเพื่อแสดงความสำเร็จ
            robot_instance.ep_led.set_led(r=0, g=255, b=0, effect="on")
            time.sleep(1)
            robot_instance.ep_led.set_led(r=0, g=0, b=255, effect="on")  # กลับเป็นสีปกติ
            
            print(f"[SHOOTING] ✅ ยิงเป้าที่ {position} เสร็จสิ้น")
            
        else:
            print("[SHOOTING] ❌ ไม่ผ่านเงื่อนไขการยิง หรือ ไม่พบเป้าหมาย")
            # รีเซ็ต gimbal กลับตำแหน่งเดิม
            robot_instance.ep_gimbal.moveto(yaw=0, pitch=0, yaw_speed=GIMBAL_TURN_SPEED).wait_for_completed()
        
    except Exception as e:
        print(f"[SHOOTING] ❌ ข้อผิดพลาด: {e}")
    
    # ทำเครื่องหมายว่ายิงแล้ว
    target_storage.mark_target_shot(position)

# =============================================================================
# DATA HANDLERS AND SENSOR MANAGEMENT
# =============================================================================

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
        """คืนค่า ToF ดิบ"""
        with self._lock:
            return self.raw_distance

class VisionDataHandler:
    """จัดการข้อมูลจากกล้อง สำหรับ marker detection"""
    
    def __init__(self):
        self.markers = []                 # [(label, x)]
        self._lock = threading.Lock()
        self._sample_logged = False       # log โครงสร้างครั้งแรกครั้งเดียว

    def update(self, vision_info):
        """อัปเดตข้อมูล marker จาก callback"""
        with self._lock:
            self.markers.clear()
            if vision_info:
                if not self._sample_logged:
                    print("[Vision raw] ->", vision_info)
                    self._sample_logged = True
                for t in vision_info:
                    # รูปแบบ: (x, y, w, h, label, ...)
                    if not isinstance(t, (list, tuple)) or len(t) < 5:
                        continue
                    x = float(t[0])  # 0..1 (ซ้าย..ขวา)
                    label = str(t[4])
                    self.markers.append((label, x))

    def get_markers(self):
        with self._lock:
            return list(self.markers)

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
        print("queue: ", queue)
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
# PLOTTING FUNCTIONS
# =============================================================================

def plot_map_with_walls(graph, blocked, path, marker_map, filename="maze_map.png"):
    """
    วาดแผนที่สุดท้ายพร้อมกำแพง เส้นทาง และ markers
    
    Args:
        graph: กราฟของเส้นทางที่เดินได้
        blocked: เส้นทางที่ถูกกำแพงปิด
        path: เส้นทางที่หุ่นเดิน
        marker_map: markers ที่พบ
        filename: ชื่อไฟล์ที่ต้องการบันทึก
    """
    plt.figure(figsize=(8, 8))
    plt.title("Robot Map with Walls and Path")

    # วาด "ทางเปิด" จาก graph (เส้นบาง)
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

    # วาด "กำแพงปิด" จาก blocked (เส้นหนา)
    for edge in blocked:
        (x1, y1), (x2, y2) = edge
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and abs(dy) == 1:
            # แนวเดียวกันคอลัมน์ (เหนือ-ใต้) -> กำแพงแนวนอนที่กึ่งกลาง
            y_mid = (y1 + y2) / 2.0
            plt.plot([x1 - 0.5, x1 + 0.5], [y_mid, y_mid], linewidth=3.0, color="k", zorder=3)
        elif dy == 0 and abs(dx) == 1:
            # แนวเดียวกันแถว (ซ้าย-ขวา) -> กำแพงแนวตั้งที่กึ่งกลาง
            x_mid = (x1 + x2) / 2.0
            plt.plot([x_mid, x_mid], [y1 - 0.5, y1 + 0.5], linewidth=3.0, color="k", zorder=3)

    # โหนด
    if graph:
        xs, ys = zip(*graph.keys())
        plt.scatter(xs, ys, s=36, color="tab:blue", zorder=2, label="Nodes")

    # เส้นทางที่เดิน
    if path:
        px, py = zip(*path)
        plt.plot(px, py, linewidth=2, color="tab:red", zorder=4, label="Path")
        plt.scatter(px[0], py[0], s=120, color="green", marker='o', zorder=5, label='Start')
        plt.scatter(px[-1], py[-1], s=120, color="purple", marker='X', zorder=5, label='End')
        
    # วาง Marker "ตามตำแหน่งจริงบนกำแพง" ด้วย offset_m (ซ้าย=ลบ, ขวา=บวก)
    for name, hits in (marker_map or {}).items():
        for finding in hits:
            (gx, gy) = finding['grid_pos']
            wall = finding['wall']
            offset = finding['offset_m'] / GRID_SIZE_M  # เป็นหน่วย "จำนวนกริด"

            if wall == "North Wall":
                # ซ้าย(ลบ) -> x ลดลง: mx = gx + offset
                mx, my = gx + offset, gy + 0.4
            elif wall == "East Wall":
                # ซ้าย(ลบ) ของผนังหันขวา = เหนือ -> y เพิ่ม: my = gy - offset
                mx, my = gx + 0.4, gy - offset
            elif wall == "South Wall":
                # ซ้าย(ลบ) ของผนังหันลง = ตะวันออก -> x เพิ่ม: mx = gx - offset
                mx, my = gx - offset, gy - 0.4
            elif wall == "West Wall":
                # ซ้าย(ลบ) ของผนังหันซ้าย = ใต้ -> y ลด: my = gy + offset
                mx, my = gx - 0.4, gy + offset
            else:
                mx, my = gx, gy

            plt.scatter([mx], [my], s=60, marker='*', color="magenta", zorder=6)
            plt.text(mx + 0.06, my + 0.06, f"{name}", fontsize=8, zorder=7)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.savefig(filename, dpi=150)
    print(f"Map saved to '{filename}'")
    plt.close()

def plot_map_with_targets(graph, blocked, path, targets, shot_targets=None, filename="maze_map_with_targets.png"):
    """วาดแผนที่พร้อมเป้าหมาย"""
    plt.figure(figsize=(8, 8))
    plt.title("Robot Path with Targets")
    drawn=set()
    for a,nbs in graph.items():
        for b in nbs:
            e=tuple(sorted([a,b]))
            if e in drawn: continue
            drawn.add(e)
            x1,y1=a; x2,y2=b
            plt.plot([x1,x2],[y1,y2],linewidth=1.0,alpha=0.35,zorder=1)
    for (a,b) in blocked:
        (x1,y1),(x2,y2)=a,b
        dx,dy=x2-x1,y2-y1
        if dx==0 and abs(dy)==1:
            y_mid=(y1+y2)/2
            plt.plot([x1-0.5,x1+0.5],[y_mid,y_mid],"k-",linewidth=2.5,zorder=3)
        elif dy==0 and abs(dx)==1:
            x_mid=(x1+x2)/2
            plt.plot([x_mid,x_mid],[y1-0.5,y1+0.5],"k-",linewidth=2.5,zorder=3)
    if path:
        px,py=zip(*path)
        plt.plot(px,py,linewidth=2.2,color="tab:red",zorder=4,label="Path")
        plt.scatter(px[0],py[0],s=110,color="green",marker='o',zorder=5,label='Start')
        plt.scatter(px[-1],py[-1],s=110,color="purple",marker='X',zorder=5,label='End')
    if targets:
        tx,ty=zip(*targets)
        plt.scatter(tx,ty,s=80,marker='*',color="magenta",zorder=6,label='Targets')
    if shot_targets:
        sx=[p[0] for p in shot_targets]; sy=[p[1] for p in shot_targets]
        plt.scatter(sx,sy,s=120,marker='X',color="black",zorder=7,label='Shot')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True,alpha=0.3); plt.legend(loc='best')
    plt.savefig(filename,dpi=150); print(f"Saved: {filename}")
    plt.close()