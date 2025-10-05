
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Function Library for RoboMaster Vision Detection
"""

import cv2
import time
import math
import numpy as np
import queue
from robomaster import robot

class PID:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, integ_clip=None, out_clip=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integ_clip = integ_clip
        self.out_clip = out_clip
        self.integ = 0.0
        self.last_error = 0.0
        self.shoot_target = []
        
    def step(self, error, dt):
        if dt <= 0:
            dt = 0.01
        
        # Proportional
        p_term = self.kp * error
        
        # Integral
        self.integ += error * dt
        if self.integ_clip:
            self.integ = max(-self.integ_clip, min(self.integ_clip, self.integ))
        i_term = self.ki * self.integ
        
        # Derivative
        d_term = self.kd * (error - self.last_error) / dt
        self.last_error = error
        
        # Output
        output = p_term + i_term + d_term
        if self.out_clip:
            output = max(-self.out_clip, min(self.out_clip, output))
        
        return output

class SimpleTofHandler:
    def __init__(self, ep_sensor):
        self.ep_sensor = ep_sensor
        
    def get_distance(self):
        try:
            return self.ep_sensor.get_distance_sensor()
        except:
            return 1000  # Default distance

class VisionDetector:
    def __init__(self, ep_robot):
        self.ep_robot = ep_robot
        self.ep_camera = ep_robot.camera
        self.ep_gimbal = ep_robot.gimbal
        self.ep_blaster = ep_robot.blaster
        self.tof_handler = SimpleTofHandler(ep_robot.sensor)
        self.last_time = time.monotonic()
        self.shoot_target = []  # เก็บรายการสิ่งที่พบทั้งหมดในรอบการตรวจจับ
        self._shot_centroids = []  # เก็บตำแหน่งที่ยิงแล้ว เพื่อตัดซ้ำ
        # สีสำหรับวาดกรอบในภาพ (BGR)
        self.draw_colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255),
            'blue': (255, 0, 0),
            'unknown': (255, 255, 255)
        }
        
        # Color ranges (HSV)
        self.color_ranges = {
            'brown_red': ([0, 120, 70], [25, 255, 255]),
            'dark_green': ([60, 40, 40], [95, 255, 255]),
            'yellow_brown': ([15, 100, 100], [25, 255, 255]),
            'dark_blue': ([90, 60, 20], [150, 255, 255])
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

    # ------------------------- Helpers -------------------------
    def _base_color(self, color_key: str) -> str:
        """แปลง color_key (เช่น 'brown_red') ไปเป็นสีพื้นฐาน ('red')"""
        for base, key in self.colors.items():
            if key == color_key:
                return base
        return color_key

    def _shape_from_contour(self, cnt) -> str:
        area = cv2.contourArea(cnt)
        if area < 1:
            return "unknown"
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            return "unknown"
        # Circle by circularity
        circularity = 4 * math.pi * (area / (peri * peri))
        if circularity > 0.75:
            return "circle"
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if h == 0:
                return "unknown"
            ratio = w / float(h)
            if 0.9 <= ratio <= 1.1:
                return "square"
            return "rectangle_h" if ratio > 1.1 else "rectangle_v"
        return "unknown"

    def _detect_all_in_frame(self, img):
        """คืนค่ารายการ detections ทั้งหมดใน frame: list[dict]"""
        detections = []
        h, w = img.shape[:2]
        roi_w, roi_h = min(800, w), min(450, h)
        roi_x, roi_y = (w - roi_w) // 2, (h - roi_h) // 2
        roi = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3, 3), np.uint8)

        for color_key, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 500:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                cx = roi_x + x + bw // 2
                cy = roi_y + y + bh // 2
                shape = self._shape_from_contour(cnt)
                detections.append({
                    "color_key": color_key,
                    "base_color": self._base_color(color_key),
                    "shape": shape,
                    "bbox": (roi_x + x, roi_y + y, bw, bh),
                    "center": (cx, cy),
                })
        return detections

    def _cluster_initial(self, dets, dist_thresh=40):
        """รวม detections ที่ซ้ำกันในหลายเฟรมโดยอิงตามสี/รูปร่าง/ตำแหน่งใกล้กัน"""
        clustered = []
        for d in dets:
            placed = False
            for c in clustered:
                # เช็คทั้งสีและตำแหน่งใกล้กัน (ไม่สนรูปร่างเพราะอาจเปลี่ยนระหว่างเฟรม)
                if d["color_key"] == c["color_key"]:
                    (cx, cy) = d["center"]
                    (ccx, ccy) = c["center"]
                    if math.hypot(cx - ccx, cy - ccy) < dist_thresh:
                        # merge โดยใช้รูปร่างที่ไม่ใช่ unknown ถ้ามี
                        if c["shape"] == "unknown" and d["shape"] != "unknown":
                            c["shape"] = d["shape"]
                        # เฉลี่ยตำแหน่ง
                        c["center"] = ((ccx + cx)//2, (ccy + cy)//2)
                        placed = True
                        break
            if not placed:
                clustered.append(d.copy())
        return clustered

    def _already_shot(self, center, tol=50):
        for (sx, sy) in self._shot_centroids:
            if math.hypot(center[0]-sx, center[1]-sy) <= tol:
                return True
        return False

    def detect_and_shot(self, sample_time=1.0, aim_timeout=6.0):
        """
        - ตรวจจับทุกรูปร่างทุกสี (แบบ aumm.py) แล้วเก็บใน self.shoot_target
        - ยิงเฉพาะวัตถุที่ตรงกับ target_color/target_shape ทีละตัว (ซ้ายไปขวา)
        - ถ้ายิงแล้วจะไม่ยิงซ้ำ
        - พิมพ์สรุป self.shoot_target หลังยิงครบ
        """
        # เตรียม
        self.shoot_target = []
        self._shot_centroids = []
        self.ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
        time.sleep(0.2)
        try:
            self.ep_gimbal.move(pitch=-15, yaw=0).wait_for_completed()
        except Exception:
            pass

        # 1) เก็บตัวอย่างหลายเฟรม เพื่อรวมเป็นชุดเป้าเริ่มต้น
        print("🔍 Scanning for objects...")
        t0 = time.time()
        pooled = []
        frame_count = 0
        while time.time() - t0 < sample_time:
            try:
                img = self.ep_camera.read_cv2_image(strategy="newest", timeout=1)
                if img is None:
                    print("⚠️ No image received from camera")
                    continue
                frame_count += 1
                print(f"📸 Processing frame {frame_count}")
            except queue.Empty:
                print("⚠️ Camera queue empty")
                continue
            dets = self._detect_all_in_frame(img)
            print(f"   Found {len(dets)} detections in frame {frame_count}")
            for i, d in enumerate(dets):
                print(f"     {i+1}. {d['base_color']} {d['shape']} at {d['center']}")
            pooled.extend(dets)
            # Visualization: draw all detections + combined binary mask of ROI
            h, w = img.shape[:2]
            roi_w, roi_h = min(800, w), min(450, h)
            roi_x, roi_y = (w - roi_w) // 2, (h - roi_h) // 2
            roi = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            combined_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            for color_key, (lower, upper) in self.color_ranges.items():
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(hsv, lower_np, upper_np)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            frame = img.copy()
            # Draw ROI rectangle
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 255), 2)
            # Draw detections
            for d in dets:
                (x, y, bw, bh) = d["bbox"]
                color_bgr = self.draw_colors.get(d["base_color"], (255, 255, 255))
                label = f"{d['base_color']}_{d['shape']}"
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color_bgr, 2)
                cx, cy = d["center"]
                cv2.circle(frame, (cx, cy), 3, color_bgr, -1)
                cv2.putText(frame, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)
            cv2.imshow("Detections", frame)
            cv2.imshow("Binary - Combined (ROI)", combined_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        initial = self._cluster_initial(pooled)
        print(f"📊 Total frames processed: {frame_count}")
        print(f"📊 Total detections: {len(pooled)}")
        print(f"📊 Clustered objects: {len(initial)}")

        # เก็บทั้งหมดลง shoot_target (ยังไม่ยิง)
        for d in initial:
            entry = {
                "color": d["base_color"],
                "color_key": d["color_key"],
                "shape": d["shape"],
                "center": d["center"],
                "bbox": d["bbox"],
                "shot": False,
            }
            self.shoot_target.append(entry)
            print(f"🎯 Added to shoot_target: {entry['color']} {entry['shape']} at {entry['center']}")

        # 2) เลือกเฉพาะที่ตรงกับ target
        # map shape synonyms
        target_shape = self.target_shape
        if target_shape == "horizontal_rectangle":
            target_shape = "rectangle_h"
        elif target_shape == "vertical_rectangle":
            target_shape = "rectangle_v"

        targets_to_shoot = [d for d in self.shoot_target
                            if d["color"] == self.target_color and d["shape"] == target_shape]
        print(f"🎯 Targets matching {self.target_color} {target_shape}: {len(targets_to_shoot)}")
        # ยิงจากซ้ายไปขวา
        targets_to_shoot.sort(key=lambda d: d["center"][0])

        # 3) ยิงทีละตัว
        if not targets_to_shoot:
            print("⚠️ No targets found matching criteria")
        else:
            print(f"🎯 Shooting {len(targets_to_shoot)} targets...")
            
        pid_yaw = PID(kp=0.20, ki=0.00, kd=0.00, integ_clip=40.0, out_clip=12.0)
        pid_pitch = PID(kp=0.20, ki=0.00, kd=0.00, integ_clip=40.0, out_clip=10.0)
        camera_w = 640  # will be corrected by first frame
        camera_h = 360

        for idx, tgt in enumerate(targets_to_shoot):
            if self._already_shot(tgt["center"]):
                print(f"🎯 Target {idx+1} already shot, skipping...")
                continue
            print(f"🎯 Aiming at target {idx+1}: {tgt['color']} {tgt['shape']} at {tgt['center']}")
            t_end = time.time() + aim_timeout
            last_seen = tgt["center"]
            aimed_successfully = False
            
            while time.time() < t_end:
                try:
                    img = self.ep_camera.read_cv2_image(strategy="newest", timeout=1)
                    if img is None:
                        continue
                except queue.Empty:
                    continue
                h, w = img.shape[:2]
                camera_w, camera_h = w//2, h//2
                dets = self._detect_all_in_frame(img)
                # Visualization during aiming
                roi_w, roi_h = min(800, w), min(450, h)
                roi_x, roi_y = (w - roi_w) // 2, (h - roi_h) // 2
                roi = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                combined_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
                for color_key, (lower, upper) in self.color_ranges.items():
                    lower_np = np.array(lower, dtype=np.uint8)
                    upper_np = np.array(upper, dtype=np.uint8)
                    mask = cv2.inRange(hsv, lower_np, upper_np)
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
                frame = img.copy()
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 255), 2)
                for d_vis in dets:
                    (x, y, bw, bh) = d_vis["bbox"]
                    color_bgr = self.draw_colors.get(d_vis["base_color"], (255, 255, 255))
                    label = f"{d_vis['base_color']}_{d_vis['shape']}"
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color_bgr, 2)
                    cx_vis, cy_vis = d_vis["center"]
                    cv2.circle(frame, (cx_vis, cy_vis), 3, color_bgr, -1)
                    cv2.putText(frame, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)
                # Draw center crosshair
                cv2.line(frame, (camera_w-20, camera_h), (camera_w+20, camera_h), (0, 0, 255), 2)
                cv2.line(frame, (camera_w, camera_h-20), (camera_w, camera_h+20), (0, 0, 255), 2)
                cv2.imshow("Detections", frame)
                cv2.imshow("Binary - Combined (ROI)", combined_mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # หา detection ใกล้พิกัดล่าสุดของเป้านี้ที่สุด
                same = [d for d in dets if d["base_color"] == tgt["color"] and d["shape"] == tgt["shape"]]
                if same:
                    # nearest
                    d = min(same, key=lambda dd: math.hypot(dd["center"][0]-last_seen[0], dd["center"][1]-last_seen[1]))
                    last_seen = d["center"]
                    cx, cy = last_seen
                    err_x = cx - camera_w
                    err_y = cy - camera_h
                    now = time.monotonic()
                    dt = now - self.last_time
                    self.last_time = now
                    yaw_cmd = pid_yaw.step(err_x, dt)
                    pitch_cmd = pid_pitch.step(err_y, dt)
                    try:
                        self.ep_gimbal.drive_speed(pitch_speed=-pitch_cmd, yaw_speed=yaw_cmd)
                    except Exception:
                        pass
                    # ถึงกลางภาพแล้ว -> ยิง
                    if abs(err_x) < 12 and abs(err_y) < 10:
                        print(f"🎯 Target centered! Firing at {last_seen}")
                        self.ep_blaster.fire(times=3)
                        time.sleep(0.6)
                        self.ep_blaster.fire(times=3)
                        time.sleep(0.6)
                        # บันทึกสถานะยิง
                        self._shot_centroids.append(last_seen)
                        tgt["shot"] = True
                        aimed_successfully = True
                        print(f"✅ Target {idx+1} shot successfully!")
                        break
                else:
                    # ไม่เห็นชั่วคราว -> ชะลอเล็กน้อย
                    time.sleep(0.05)
            
            if not aimed_successfully:
                print(f"⚠️ Failed to aim at target {idx+1} within timeout")
                
            # หยุดกิมบอลเล็กน้อยก่อนเป้าถัดไป
            try:
                self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception:
                pass
            time.sleep(0.3)

        # 4) แสดงสรุป
        shot_count = sum(1 for s in self.shoot_target if s["shot"])
        total_targets = len([s for s in self.shoot_target if s["color"] == self.target_color and 
                            (s["shape"] == target_shape or 
                             (target_shape == "rectangle_h" and s["shape"] == "horizontal_rectangle") or
                             (target_shape == "rectangle_v" and s["shape"] == "vertical_rectangle"))])
        
        print(f"\n🎯 Mission Summary: Shot {shot_count}/{total_targets} {self.target_color} {self.target_shape} targets")
        print("\n[Summary] shoot_target:")
        for i, s in enumerate(self.shoot_target, 1):
            status = "✅ SHOT" if s["shot"] else "❌ MISSED"
            print(f"  {i}. color={s['color']} ({s['color_key']}), shape={s['shape']}, center={s['center']}, {status}")
        try:
            cv2.destroyWindow("Binary - Combined (ROI)")
            cv2.destroyWindow("Detections")
        except Exception:
            pass
        return self.shoot_target
# Test function
def test_camera_only():
    """ทดสอบกล้องเพียงอย่างเดียว"""
    print("🤖 Initializing RoboMaster...")
    
    # เชื่อมต่อหุ่นยนต์
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    try:
        # สร้าง detector
        detector = VisionDetector(ep_robot)
        
        # เริ่ม camera stream
        print("📷 Starting camera stream...")
        detector.ep_camera.start_video_stream(display=False, resolution="720p")
        time.sleep(2)
        print("✅ Camera stream started")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            try:
                # อ่านภาพ
                img = detector.ep_camera.read_cv2_image(strategy="newest", timeout=1)
                if img is None:
                    print("⚠️ No image received")
                    continue
                
                # แสดงข้อมูล
                frame_count += 1
                fps = frame_count / (time.time() - start_time)
                height, width = img.shape[:2]
                
                cv2.putText(img, f"Camera Test - Frame: {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"Resolution: {width}x{height}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"FPS: {fps:.1f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, "Press 'q' to quit", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # วาดกรอบกลาง
                center_x, center_y = width // 2, height // 2
                cv2.circle(img, (center_x, center_y), 20, (0, 0, 255), 2)
                cv2.line(img, (center_x-50, center_y), (center_x+50, center_y), (0, 0, 255), 2)
                cv2.line(img, (center_x, center_y-50), (center_x, center_y+50), (0, 0, 255), 2)
                
                cv2.imshow("RoboMaster Camera Test", img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"⚠️ Error: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        print("🧹 Cleaning up...")
        try:
            detector.ep_camera.stop_video_stream()
        except:
            pass
        cv2.destroyAllWindows()
        ep_robot.close()
        print("✅ Test completed")

def test_vision_detection():
    """ทดสอบระบบตรวจจับวัตถุ"""
    print("🤖 Initializing RoboMaster for vision test...")
    
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    try:
        # สร้าง detector
        detector = VisionDetector(ep_robot)
        
        # เริ่ม camera stream
        print("📷 Starting camera stream...")
        detector.ep_camera.start_video_stream(display=False, resolution="720p")
        time.sleep(2)
        print("✅ Camera stream started")
        
        # ตั้งค่าเป้าหมาย (ตัวอย่าง: หาวงกลมสีน้ำเงิน)
        """
        รันการตรวจจับวัตถุด้วยกล้อง RoboMaster แบบสั้นๆ
        
        target_color: สีเป้าหมาย ('red', 'green', 'blue', 'yellow')
        target_shape: รูปทรงเป้าหมาย
            - 'circle': วงกลม
            - 'square': สี่เหลี่ยมจัตุรัส  
            - 'vertical_rectangle': สี่เหลี่ยมผืนผ้าแนวตั้ง
            - 'horizontal_rectangle': สี่เหลี่ยมผืนผ้าแนวนอน
        """
        detector.set_target('green', 'square')
        
        print(f"🎯 Looking for: {detector.target_color} {detector.target_shape}")
        
        # ทดสอบ detection
        detector.detect_and_shot()
        
    except Exception as e:
        print(f"❌ Vision test failed: {e}")
    finally:
        try:
            detector.ep_camera.stop_video_stream()
        except:
            pass
        cv2.destroyAllWindows()
        ep_robot.close()

if __name__ == "__main__":
    print("=" * 50)
    print("🎥 RoboMaster Function Test")
    print("=" * 50)
    print("1. Camera only test")
    print("2. Vision detection test")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_camera_only()
    elif choice == "2":
        test_vision_detection()
    else:
        print("Invalid choice. Running camera test...")
        test_camera_only()