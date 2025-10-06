
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
        self._fired_positions = []  # เก็บตำแหน่งที่ยิงแล้ว เพื่อตัดซ้ำ
        
        # ⚙️ การตั้งค่า ROI สำหรับ State 2 (ปรับได้)
        self.state2_roi_config = {
            'width': 450,           # ความกว้าง ROI
            'height': 450,          # ความสูง ROI  
            'offset_x': 0,          # เลื่อนซ้าย(-)/ขวา(+) จากกึ่งกลาง
            'offset_y': 0,          # เลื่อนบน(-)/ล่าง(+) จากกึ่งกลาง
            'use_dynamic': False,   # ใช้ ROI แบบไดนามิกตามเป้าหมาย
            'dynamic_size': 300     # ขนาด ROI แบบไดนามิก
        }
        # สีสำหรับวาดกรอบในภาพ (BGR)
        self.draw_colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255),
            'blue': (255, 0, 0),
            'unknown': (255, 255, 255)
        }
        
        # Color ranges (HSV) - รองรับ multiple ranges สำหรับสีแดง
        self.color_ranges = {
            # red ต้อง 2 ช่วง: 0–10 และ 170–180 (tuple ของคู่ช่วงสี)
            'brown_red': (([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])),
            'dark_green': ([40, 60, 20], [85, 255, 180]),
            'yellow_brown': ([18, 100, 100], [40, 255, 255]),
            'dark_blue': ([95, 70, 20], [130, 255, 200])
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
    
    def set_state2_roi(self, width=550, height=300, offset_x=0, offset_y=0, use_dynamic=False, dynamic_size=300):
        """🎯 ตั้งค่า ROI สำหรับ State 2
        Args:
            width: ความกว้าง ROI (default: 450)
            height: ความสูง ROI (default: 450)  
            offset_x: เลื่อนซ้าย(-)/ขวา(+) จากกึ่งกลาง (default: 0)
            offset_y: เลื่อนบน(-)/ล่าง(+) จากกึ่งกลาง (default: 0)
            use_dynamic: ใช้ ROI แบบไดนามิกตามเป้าหมาย (default: False)
            dynamic_size: ขนาด ROI แบบไดนามิก (default: 300)
        """
        self.state2_roi_config.update({
            'width': width,
            'height': height,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'use_dynamic': use_dynamic,
            'dynamic_size': dynamic_size
        })
        print(f"🎯 State 2 ROI updated: {width}x{height}, offset({offset_x},{offset_y}), dynamic={use_dynamic}")
    
    def _get_state2_roi(self, img, target_center=None):
        """คำนวณ ROI สำหรับ State 2"""
        h, w = img.shape[:2]
        config = self.state2_roi_config
        
        if config['use_dynamic'] and target_center is not None:
            # ROI แบบไดนามิกตามตำแหน่งเป้าหมาย
            size = config['dynamic_size']
            target_x, target_y = target_center
            roi_x = max(0, min(w - size, target_x - size//2))
            roi_y = max(0, min(h - size, target_y - size//2))
            roi_w = min(size, w - roi_x)
            roi_h = min(size, h - roi_y)
        else:
            # ROI แบบคงที่ที่ปรับได้
            roi_w = min(config['width'], w)
            roi_h = min(config['height'], h)
            
            # คำนวณตำแหน่งพร้อม offset
            center_x = w // 2 + config['offset_x']
            center_y = h // 2 + config['offset_y']
            
            roi_x = max(0, min(w - roi_w, center_x - roi_w//2))
            roi_y = max(0, min(h - roi_h, center_y - roi_h//2))
        
        return roi_x, roi_y, roi_w, roi_h
    
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

        for color_key, color_range in self.color_ranges.items():
            # รองรับ multiple ranges สำหรับสี (เช่น สีแดงมี 2 ช่วง)
            if isinstance(color_range[0][0], (list, tuple)):
                # Multiple ranges (เช่น brown_red)
                combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                for lower, upper in color_range:
                    lower = np.array(lower, dtype=np.uint8)
                    upper = np.array(upper, dtype=np.uint8)
                    range_mask = cv2.inRange(hsv, lower, upper)
                    combined_mask = cv2.bitwise_or(combined_mask, range_mask)
                mask = combined_mask
            else:
                # Single range (เช่น dark_green, yellow_brown, dark_blue)
                lower, upper = color_range
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)
            
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # เพิ่มเงื่อนไขการกรองขนาดวัตถุให้เข้มงวดขึ้น
                if area < 800:  # เพิ่มจาก 500 เป็น 800
                    continue
                    
                x, y, bw, bh = cv2.boundingRect(cnt)
                
                # กรองวัตถุที่เล็กเกินไปในแต่ละด้าน
                if bw < 20 or bh < 20:
                    continue
                    
                # กรองวัตถุที่มี aspect ratio แปลกๆ
                aspect_ratio = max(bw, bh) / min(bw, bh)
                if aspect_ratio > 5:  # ถ้ายาวหรือสูงเกินไป
                    continue
                
                cx = roi_x + x + bw // 2
                cy = roi_y + y + bh // 2
                shape = self._shape_from_contour(cnt)
                
                # เพิ่มข้อมูลขนาดเพื่อช่วยในการตัดสินใจ
                detections.append({
                    "color_key": color_key,
                    "base_color": self._base_color(color_key),
                    "shape": shape,
                    "bbox": (roi_x + x, roi_y + y, bw, bh),
                    "center": (cx, cy),
                    "area": area,
                    "confidence": min(area / 1000.0, 1.0)  # confidence จากขนาด
                })
        return detections

    def _cluster_initial(self, dets, dist_thresh=40):
        """รวม detections ที่ซ้ำกันในหลายเฟรมแบบอัจฉริยะ"""
        clustered = []
        
        for d in dets:
            placed = False
            for c in clustered:
                # เช็คทั้งสีและตำแหน่งใกล้กัน
                if d["color_key"] == c["color_key"]:
                    (cx, cy) = d["center"]
                    (ccx, ccy) = c["center"]
                    if math.hypot(cx - ccx, cy - ccy) < dist_thresh:
                        # เลือกรูปร่างที่ชัดเจนกว่า (ไม่ใช่ unknown)
                        if c["shape"] == "unknown" and d["shape"] != "unknown":
                            c["shape"] = d["shape"]
                        elif c["shape"] != "unknown" and d["shape"] == "unknown":
                            pass  # ใช้รูปร่างเดิม
                        elif c["shape"] != d["shape"] and c["shape"] != "unknown" and d["shape"] != "unknown":
                            # ถ้ารูปร่างไม่ตรงกัน ใช้ที่มี confidence สูงกว่า
                            if d.get("confidence", 0) > c.get("confidence", 0):
                                c["shape"] = d["shape"]
                        
                        # เฉลี่ยตำแหน่งแบบถ่วงน้ำหนัก
                        old_conf = c.get("confidence", 0.5)
                        new_conf = d.get("confidence", 0.5)
                        total_conf = old_conf + new_conf
                        
                        if total_conf > 0:
                            c["center"] = (
                                int((ccx * old_conf + cx * new_conf) / total_conf),
                                int((ccy * old_conf + cy * new_conf) / total_conf)
                            )
                        
                        # เพิ่ม confidence และ detection count
                        c["confidence"] = min(c.get("confidence", 0) + d.get("confidence", 0), 2.0)
                        c["detection_count"] = c.get("detection_count", 1) + 1
                        c["area"] = max(c.get("area", 0), d.get("area", 0))  # ใช้ area ที่ใหญ่กว่า
                        
                        placed = True
                        break
            
            if not placed:
                new_cluster = d.copy()
                new_cluster["detection_count"] = 1
                clustered.append(new_cluster)
        
        # กรองเฉพาะวัตถุที่มี confidence สูงและถูกตรวจจับหลายครั้ง
        reliable_objects = []
        for c in clustered:
            confidence = c.get("confidence", 0)
            count = c.get("detection_count", 1)
            area = c.get("area", 0)
            
            # เงื่อนไขความน่าเชื่อถือ
            if (confidence >= 0.8 and count >= 2) or (confidence >= 1.2 and count >= 1):
                if area >= 800:  # ขนาดขั้นต่ำ
                    reliable_objects.append(c)
        
        # เรียงตามความน่าเชื่อถือ (confidence * detection_count)
        reliable_objects.sort(key=lambda x: x.get("confidence", 0) * x.get("detection_count", 1), reverse=True)
        
        # จำกัดจำนวนวัตถุไม่เกิน 5 อัน (ป้องกันการตรวจจับผิด)
        return reliable_objects[:5]

    def _already_fired(self, center, tol=50):
        for (sx, sy) in self._fired_positions:
            if math.hypot(center[0]-sx, center[1]-sy) <= tol:
                return True
        return False

    def detect_and_fire(self, sample_time=1.0, aim_timeout=6.0):
        """
        STEP 1: สแกนและบันทึกวัตถุทั้งหมดจากซ้ายไปขวา
        STEP 2: หาและยิงเฉพาะวัตถุที่ต้องการจากซ้ายไปขวา แล้วหยุด
        """
        # เตรียมระบบ
        self.shoot_target = []
        self._fired_positions = []
        self.ep_gimbal.recenter(pitch_speed=100, yaw_speed=100).wait_for_completed()
        time.sleep(0.2)
        try:
            self.ep_gimbal.move(pitch=-15, yaw=0).wait_for_completed()
        except Exception:
            pass

        # STEP 1: สแกนและบันทึกวัตถุทั้งหมด (แบบละเอียด)
        print("🔍 STEP 1: Advanced scanning for reliable object detection...")
        t0 = time.time()
        pooled = []
        frame_count = 0
        
        # เพิ่มเวลาในการสแกนเพื่อความแม่นยำ
        extended_sample_time = max(sample_time, 2.0)  # อย่างน้อย 2 วินาที
        
        while time.time() - t0 < extended_sample_time:
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
            for color_key, color_range in self.color_ranges.items():
                # รองรับ multiple ranges สำหรับสี
                if isinstance(color_range[0][0], (list, tuple)):
                    # Multiple ranges (เช่น brown_red)
                    for lower, upper in color_range:
                        lower_np = np.array(lower, dtype=np.uint8)
                        upper_np = np.array(upper, dtype=np.uint8)
                        mask = cv2.inRange(hsv, lower_np, upper_np)
                        combined_mask = cv2.bitwise_or(combined_mask, mask)
                else:
                    # Single range
                    lower, upper = color_range
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
        # จัดกลุ่มและจัดเรียงตามความน่าเชื่อถือ
        initial = self._cluster_initial(pooled)
        # ไม่เรียงตามตำแหน่ง ใช้ลำดับความน่าเชื่อถือจาก _cluster_initial
        
        print(f"📊 Total frames processed: {frame_count}")
        print(f"📊 Total detections: {len(pooled)}")
        print(f"📊 Clustered objects: {len(initial)}")

        # บันทึกวัตถุทั้งหมดจากซ้ายไปขวา พร้อมข้อมูลความน่าเชื่อถือ
        print("\n💾 Saving reliable objects (by confidence):")
        for i, d in enumerate(initial):
            entry = {
                "color": d["base_color"],
                "color_key": d["color_key"],
                "shape": d["shape"],
                "center": d["center"],
                "bbox": d["bbox"],
                "completed": False,
                "confidence": d.get("confidence", 0),
                "detection_count": d.get("detection_count", 1),
                "area": d.get("area", 0)
            }
            self.shoot_target.append(entry)
            conf_str = f"conf:{entry['confidence']:.1f} count:{entry['detection_count']} area:{entry['area']}"
            print(f"  {i+1}. {entry['color']} {entry['shape']} at {entry['center']} [{conf_str}]")

        # STEP 2: หาและยิงเฉพาะวัตถุที่ต้องการ
        print(f"\n🎯 STEP 2: Looking for target: {self.target_color} {self.target_shape}")
        
        # แปลงชื่อรูปร่าง
        target_shape = self.target_shape
        if target_shape == "horizontal_rectangle":
            target_shape = "rectangle_h"
        elif target_shape == "vertical_rectangle":
            target_shape = "rectangle_v"

        # หาวัตถุที่ตรงกับเป้าหมาย (เรียงตามลำดับที่เก็บไว้)
        targets_to_shoot = [d for d in self.shoot_target
                            if d["color"] == self.target_color and d["shape"] == target_shape]
        
        print(f"🎯 Found {len(targets_to_shoot)} matching targets:")
        for i, tgt in enumerate(targets_to_shoot):
            print(f"  Target {i+1}: {tgt['color']} {tgt['shape']} at {tgt['center']}")

        # เริ่มยิงเป้าหมาย (ตามลำดับที่เก็บไว้)
        if not targets_to_shoot:
            print("⚠️ No matching targets found!")
            return self.shoot_target
        
        print(f"\n🔫 Starting to shoot {len(targets_to_shoot)} targets (in stored order)...")
        
        # ตั้งค่า PID controller
        pid_yaw = PID(kp=0.20, ki=0.00, kd=0.00, integ_clip=40.0, out_clip=12.0)
        pid_pitch = PID(kp=0.20, ki=0.00, kd=0.00, integ_clip=40.0, out_clip=10.0)
        camera_w = 640  # จะถูกปรับจากเฟรมแรก
        camera_h = 360

        # ยิงทีละตัวตามลำดับที่เก็บไว้
        targets_remaining = targets_to_shoot.copy()  # สำเนารายการเป้าหมาย
        target_index = 0
        
        while targets_remaining and target_index < len(targets_remaining):
            tgt = targets_remaining[target_index]
            if self._already_fired(tgt["center"]):
                print(f"🎯 Target {target_index+1} already fired, skipping...")
                targets_remaining.pop(target_index)  # ลบเป้าหมายที่ยิงแล้วออกจากรายการ
                continue
                
            print(f"\n🎯 Engaging Target {target_index+1}/{len(targets_remaining)}: {tgt['color']} {tgt['shape']} at {tgt['center']}")
            
            # ขั้นตอนที่ 1: Recenter กิมบาล
            print(f"🔄 Recentering gimbal...")
            try:
                self.ep_gimbal.recenter().wait_for_completed()
                time.sleep(0.5)
                print(f"✅ Gimbal recentered")
            except Exception as e:
                print(f"⚠️ Recenter failed: {e}")
            
            # ขั้นตอนที่ 2: หันไปตำแหน่งเป้าหมายทันที (ไม่ใช้ฟังก์ชันความน่าเชื่อถือ)
            print(f"📍 Direct aiming to stored position {tgt['center']}...")
            target_x, target_y = tgt["center"]
            
            # คำนวณมุมที่ต้องหันไป (สมมติกล้องอยู่กลางภาพ)
            current_img = None
            try:
                current_img = self.ep_camera.read_cv2_image(strategy="newest", timeout=1)
            except:
                pass
                
            if current_img is not None:
                h, w = current_img.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                # คำนวณการเคลื่อนที่ที่ต้องการ
                delta_x = target_x - center_x
                delta_y = target_y - center_y
                
                # หันไปทางเป้าหมายอย่างรวดเร็ว (ประมาณ 70% ของระยะทาง)
                if abs(delta_x) > 30 or abs(delta_y) > 30:
                    # ใช้ moveto สำหรับการเคลื่อนที่แบบรวดเร็ว
                    yaw_move = delta_x * 0.10   # เพิ่มความไวสำหรับการหันทันที
                    pitch_move = -delta_y * 0.08  # เพิ่มความไวสำหรับการหันทันที
                    
                    # จำกัดการเคลื่อนที่
                    yaw_move = max(-20, min(20, yaw_move))
                    pitch_move = max(-15, min(15, pitch_move))
                    
                    print(f"  🔄 Direct turn to target: yaw={yaw_move:.1f}°, pitch={pitch_move:.1f}°")
                    try:
                        self.ep_gimbal.move(yaw=yaw_move, pitch=pitch_move).wait_for_completed()
                        time.sleep(0.3)
                        print(f"  ✅ Direct turn completed")
                    except Exception as e:
                        print(f"  ⚠️ Direct turn failed: {e}")
            
            # ขั้นตอนที่ 3: เข้าสู่ฟังก์ชันความน่าเชื่อถือ (PID Fine Aiming)
            print(f"🎯 Entering reliability function - Fine aiming with PID...")
            aim_timeout = 8.0  # กำหนดเวลา timeout สำหรับการเล็ง
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
                # Visualization during aiming - ใช้ ROI ที่กำหนดได้สำหรับ State 2
                roi_x, roi_y, roi_w, roi_h = self._get_state2_roi(img, tgt["center"])
                roi = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                combined_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
                for color_key, color_range in self.color_ranges.items():
                    # รองรับ multiple ranges สำหรับสี
                    if isinstance(color_range[0][0], (list, tuple)):
                        # Multiple ranges (เช่น brown_red)
                        for lower, upper in color_range:
                            lower_np = np.array(lower, dtype=np.uint8)
                            upper_np = np.array(upper, dtype=np.uint8)
                            mask = cv2.inRange(hsv, lower_np, upper_np)
                            combined_mask = cv2.bitwise_or(combined_mask, mask)
                    else:
                        # Single range
                        lower, upper = color_range
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
                # หา detection ที่ตรงกับเป้าหมายทั้งหมด
                same = [d for d in dets if d["base_color"] == tgt["color"] and d["shape"] == tgt["shape"]]
                if same:
                    # เลือกเป้าหมายที่ใกล้ที่สุดกับตำแหน่งล่าสุด
                    d = min(same, key=lambda dd: math.hypot(dd["center"][0]-last_seen[0], dd["center"][1]-last_seen[1]))
                    last_seen = d["center"]
                    cx, cy = last_seen
                    err_x = cx - camera_w
                    err_y = cy - camera_h
                    
                    # แสดงข้อมูล error สำหรับ debug
                    if abs(err_x) > 5 or abs(err_y) > 5:
                        print(f"    🎯 Adjusting: err_x={err_x:3.0f}, err_y={err_y:3.0f}")
                    
                    now = time.monotonic()
                    dt = now - self.last_time
                    self.last_time = now
                    yaw_cmd = pid_yaw.step(err_x, dt)
                    pitch_cmd = pid_pitch.step(err_y, dt)
                    
                    try:
                        self.ep_gimbal.drive_speed(pitch_speed=-pitch_cmd, yaw_speed=yaw_cmd)
                    except Exception:
                        pass
                    
                    # เล็งได้แล้ว -> ยิง (ลดความแม่นยำเล็กน้อยเพื่อให้ยิงได้ง่ายขึ้น)
                    if abs(err_x) < 15 and abs(err_y) < 12:
                        print(f"🔥 Target centered! Firing at {last_seen} (err_x={err_x:.0f}, err_y={err_y:.0f})")
                        
                        # ยิงหลายครั้งเพื่อความแน่ใจ
                        self.ep_blaster.fire(times=3)
                        time.sleep(0.3)
                        self.ep_blaster.fire(times=2)
                        time.sleep(0.3)
                        
                        # บันทึกสถานะยิง
                        self._fired_positions.append(last_seen)
                        tgt["completed"] = True
                        aimed_successfully = True
                        print(f"✅ Target {target_index+1} fired successfully!")
                        break
                else:
                    # ไม่เห็นเป้า -> รอสักครู่
                    time.sleep(0.05)
            
            if not aimed_successfully:
                print(f"⚠️ Failed to shoot target {target_index+1} within {aim_timeout}s timeout")
                
            # หยุดกิมบาลก่อนไปเป้าถัดไป
            try:
                self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            except Exception:
                pass
            time.sleep(0.3)
            
            # ขั้นตอนที่ 4: ลบข้อมูลเป้าหมายที่ยิงสำเร็จออกจากรายการ
            if aimed_successfully:
                print(f"🗑️ Removing target {target_index+1} from list...")
                targets_remaining.pop(target_index)
                # ไม่เพิ่ม target_index เพราะรายการสั้นลงแล้ว
                
                # Recenter หลังจากยิงสำเร็จ
                print(f"🔄 Post-firing recenter...")
                try:
                    self.ep_gimbal.recenter().wait_for_completed()
                    time.sleep(0.5)
                    print(f"✅ Post-firing recenter completed")
                except Exception as e:
                    print(f"⚠️ Post-firing recenter failed: {e}")
            else:
                # ถ้ายิงไม่สำเร็จ ไปเป้าถัดไป
                target_index += 1
            
            # แสดงความคืบหน้า
            completed_so_far = sum(1 for s in self.shoot_target if s["completed"])
            print(f"📊 Progress: Completed {completed_so_far} targets, {len(targets_remaining)} targets remaining")
        
        print(f"\n🏁 STEP 2 COMPLETED: Finished shooting sequence!")

        # สรุปผลการปฏิบัติงาน
        completed_count = sum(1 for s in self.shoot_target if s["completed"])
        total_found = len(self.shoot_target)
        target_count = len(targets_to_shoot)
        
        print(f"\n" + "="*60)
        print(f"🎯 MISSION SUMMARY")
        print(f"="*60)
        print(f"📊 Total objects found: {total_found}")
        print(f"🎯 Target objects ({self.target_color} {self.target_shape}): {target_count}")
        print(f"🔫 Successfully fired: {completed_count}/{target_count}")
        print(f"📈 Success rate: {(completed_count/target_count*100):.1f}%" if target_count > 0 else "📈 Success rate: N/A")
        
        print(f"\n📝 DETAILED RESULTS (by reliability):")
        for i, s in enumerate(self.shoot_target, 1):
            is_target = (s["color"] == self.target_color and s["shape"] == target_shape)
            target_marker = "🎯" if is_target else "  "
            status = "✅ COMPLETED" if s["completed"] else ("❌ MISSED" if is_target else "⚪ IGNORED")
            print(f"  {target_marker} {i:2d}. {s['color']:6s} {s['shape']:12s} at {s['center']} | {status}")
        
        # ปิดหน้าต่าง
        try:
            cv2.destroyWindow("Binary - Combined (ROI)")
            cv2.destroyWindow("Detections")
        except Exception:
            pass
            
        print(f"\n🏁 DETECTION AND SHOOTING COMPLETED!")
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
        detector.set_target('red', 'vertical_rectangle')
        
        # 🎯 ตัวอย่างการปรับ ROI สำหรับ State 2
        # detector.set_state2_roi(width=600, height=400, offset_x=50, offset_y=-30)  # ROI ขนาด 600x400 เลื่อนขวา 50px บน 30px
        # detector.set_state2_roi(width=300, height=300, use_dynamic=True, dynamic_size=350)  # ROI แบบไดนามิกตามเป้าหมาย
        
        print(f"🎯 Looking for: {detector.target_color} {detector.target_shape}")
        print(f"📐 State 2 ROI: {detector.state2_roi_config}")
        
        # ทดสอบ detection พร้อมแสดงภาพ
        detector.detect_and_fire()
        
        # แสดงภาพจากกล้องต่อเนื่องหลังจากยิงเสร็จ
        print("\n📹 Press 'q' to quit camera view...")
        while True:
            try:
                img = detector.ep_camera.read_cv2_image(strategy="newest", timeout=1)
                if img is not None:
                    # แสดงข้อมูลบนภาพ
                    height, width = img.shape[:2]
                    cv2.putText(img, "Camera Live View - Press 'q' to quit", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, f"Resolution: {width}x{height}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # วาดกรอบกลาง
                    center_x, center_y = width // 2, height // 2
                    cv2.circle(img, (center_x, center_y), 10, (0, 0, 255), 2)
                    cv2.line(img, (center_x-30, center_y), (center_x+30, center_y), (0, 0, 255), 2)
                    cv2.line(img, (center_x, center_y-30), (center_x, center_y+30), (0, 0, 255), 2)
                    
                    cv2.imshow("RoboMaster Live Camera", img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                time.sleep(0.1)
                continue
        
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