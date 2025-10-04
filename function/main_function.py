
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Function Library for RoboMaster Vision Detection
"""

import cv2
import time
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
        
        # Color ranges (HSV)
        self.color_ranges = {
            'brown_red': ([0, 120, 70], [15, 255, 255]),
            'dark_green': ([70, 40, 40], [80, 255, 255]),
            'yellow_brown': ([15, 100, 100], [25, 255, 255]),
            'dark_blue': ([110, 60, 20], [135, 255, 255]),
            'dark_purple_red': ([140, 100, 50], [180, 255, 255]),
            'vivid_dark_red': ([0, 150, 100], [10, 255, 255])
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

    def detect_and_shot(self):
        self.ep_gimbal.recenter().wait_for_completed()
        pid_yaw = PID(kp=0.20, ki=0.00, kd=0.00, integ_clip=40.0, out_clip=10.0)
        pid_pitch = PID(kp=0.20, ki=0.00, kd=0.00, integ_clip=40.0, out_clip=8.0)
        center_x = center_y = None
        err_x = err_y = 0.0
        yaw_cmd = pitch_cmd = 0.0
        # ensure camera stream is running
        try:
            self.ep_camera.start_video_stream(display=False, resolution="720p")
        except Exception as e:
            print(f"❌ Failed to start camera stream: {e}")
            return
        have_object = False  # เพิ่มการประกาศตัวแปร
        time.sleep(0.5)
        timebefore = time.monotonic()
        while True:
            try:
                img = self.ep_camera.read_cv2_image(strategy="newest", timeout=1)
                if img is None:
                    continue
            except queue.Empty:
                continue
            # แปลงชื่อสีเก่าเป็นใหม่ (ถ้าจำเป็น)
            color_key = self.target_color
            if self.target_color in self.colors:
                color_key = self.colors[self.target_color]
            
            # ตรวจสอบว่ามีสีที่ต้องการหรือไม่
            if color_key not in self.color_ranges:
                print(f"⚠️ ไม่พบสี '{self.target_color}' ในระบบ")
                return []
            # ROI (กลางภาพ)
            h, w = img.shape[:2]
            roi_w, roi_h = min(800, w), min(450, h)
            roi_x, roi_y = (w - roi_w) // 2, (h - roi_h) // 2
            roi = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            
            # แปลง HSV และสร้าง mask (เพิ่มการเบลอ + morphology เพื่อลด noise)
            blur = cv2.GaussianBlur(roi, (5, 5), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            color_data = self.color_ranges[color_key]
            
            # สร้าง mask จากช่วงสี
            lower_bound = np.array(color_data[0])
            upper_bound = np.array(color_data[1])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            # หา contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                
                if cv2.contourArea(cnt) < 200:
                    continue
                
                # ตรวจจับรูปทรง
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                vertices = len(approx)
                
                shape = None
                if vertices >= 8:
                    area = cv2.contourArea(cnt)
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.7:
                        shape = 'circle'
                elif vertices == 3:
                    shape = 'triangle'
                elif vertices == 4:
                    # ใช้ bounding box เพื่อแยกประเภทสี่เหลี่ยม
                    x, y, w_box, h_box = cv2.boundingRect(cnt)
                    aspect_ratio = float(w_box) / h_box
                    
                    if 0.9 <= aspect_ratio <= 1.1:
                        shape = 'square'  # สี่เหลี่ยมจัตุรัส
                    elif aspect_ratio > 1.3:
                        shape = 'horizontal_rectangle'  # สี่เหลี่ยมผืนผ้าแนวนอน (กว้างมากกว่าสูง)
                    elif aspect_ratio < 0.7:
                        shape = 'vertical_rectangle'  # สี่เหลี่ยมผืนผ้าแนวตั้ง (สูงมากกว่ากว้าง)
                    else:
                        shape = 'rectangle'  # สี่เหลี่ยมทั่วไป
                
                if shape == self.target_shape:
                    x, y, w_box, h_box = cv2.boundingRect(cnt)
                    center_x = roi_x + x + w_box // 2
                    center_y = roi_y + y + h_box // 2
                    break  # เจอแล้วออกจาก loop

            # วาดกรอบและข้อความ (ถ้าเจอวัตถุ)
            if center_x is not None and center_y is not None:
                cv2.rectangle(img, (center_x-30, center_y-30), (center_x+30, center_y+30), (0, 255, 0), 2)
                cv2.putText(img, f"{self.target_shape}", (center_x-40, center_y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        
            # แสดงข้อมูล
            cv2.putText(img, f"Detecting: {self.target_color} {self.target_shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"Found: {1 if center_x is not None and center_y is not None else 0} objects", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "Press 'q' to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("RoboMaster Detection", img)
            # แสดงหน้าต่าง mask เพื่อ debug สี (ช่วยหาช่วง HSV)
            try:
                cv2.imshow("Mask", mask)
            except Exception:
                pass
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🔴 User pressed 'q' - exiting detection")
                break
            
            now = time.monotonic()
            dt = now - self.last_time
            self.last_time = now

            camera_w = 1280//2
            camera_h = 720//2
            print(self.tof_handler.get_distance())
            
            if center_x is not None and center_y is not None and self.tof_handler.get_distance() < 1200:
                have_object = True
                err_x = (center_x - camera_w)
                err_y = (center_y - camera_h)
                yaw_cmd = pid_yaw.step(err_x, dt)
                pitch_cmd = pid_pitch.step(err_y, dt)
                
                if abs(yaw_cmd) > 0.05 or abs(pitch_cmd) > 0.05:
                    try:
                        print(1)
                        self.ep_gimbal.drive_speed(pitch_speed=-pitch_cmd, yaw_speed=yaw_cmd)
                    except Exception:
                        try:
                            self.ep_gimbal.drive_speed(pitch_speed=pitch_cmd, yaw_speed=yaw_cmd)
                        except Exception:
                            pass
                    
            elif time.monotonic() - timebefore < 10.0 and have_object:
                # ยังอยู่ในช่วงเวลาและเคยเจอวัตถุ - หยุดการเคลื่อนไหว gimbal
                self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
                
                print(1)
                pid_yaw.integ *= 0.9
                pid_pitch.integ *= 0.9
                
                
            # รีเซ็ต center_x, center_y สำหรับ frame ถัดไป
            center_x = center_y = None

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
        detector.set_target('red', 'horizontal_rectangle')
        
        print(f"🎯 Looking for: {detector.target_color} {detector.target_shape}")
        
        # ทดสอบ detection
        detector.detect_and_shot()
        
    except Exception as e:
        print(f"❌ Vision test failed: {e}")
    finally:
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