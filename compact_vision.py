"""
Compact Vision Detection for Project_2
=====================================
ใช้สำหรับตรวจจับรูปทรงและสีในขั้นตอน scan หลังจากสแกนกำแพงเสร็จ
"""

import cv2
import numpy as np
import time
import math
import time
from robomaster import robot
class CompactVisionDetector:
    """ระบบตรวจจับรูปทรงและสีแบบย่อ"""
    
    def __init__(self):
        # กำหนดช่วงสี HSV และสีสำหรับการแสดงผล
        self.color_ranges = {
            "brown_red":      ([0, 150, 50],   [10, 255, 120],  (0, 0, 255)),
            "dark_green":     ([70, 150, 20],  [90, 255, 70],   (0, 255, 0)),
            "yellow_brown":   ([15, 150, 80],  [30, 255, 180],  (0, 255, 255)),
            "dark_blue":      ([110, 150, 30], [130, 255, 100], (255, 0, 0)),
            "dark_purple_red":([170, 150, 50], [180, 255, 150], (255, 0, 255)),
            "vivid_dark_red": ([170, 180, 50], [180, 255, 200], (255, 255, 255))
        }
        
        # สำหรับ backward compatibility (แมปสีเก่าไปใหม่)
        self.colors = {
            'red': "brown_red",
            'green': "dark_green", 
            'blue': "dark_blue",
            'yellow': "yellow_brown",
            'purple': "dark_purple_red",
            'dark_red': "vivid_dark_red"
        }
        
        self.target_color = None
        self.target_shape = None
    
    def set_target(self, color, shape):
        """ตั้งค่าเป้าหมาย"""
        self.target_color = color
        self.target_shape = shape
    
    def detect(self, frame):
        """ตรวจจับเป้าหมาย - คืนค่า [(center_x, center_y), ...] หรือ []"""
        if not self.target_color or not self.target_shape:
            return []
        
        # แปลงชื่อสีเก่าเป็นใหม่ (ถ้าจำเป็น)
        color_key = self.target_color
        if self.target_color in self.colors:
            color_key = self.colors[self.target_color]
        
        # ตรวจสอบว่ามีสีที่ต้องการหรือไม่
        if color_key not in self.color_ranges:
            print(f"⚠️ ไม่พบสี '{self.target_color}' ในระบบ")
            return []
        
        # ROI (กลางภาพ)
        h, w = frame.shape[:2]
        roi_w, roi_h = min(800, w), min(450, h)
        roi_x, roi_y = (w - roi_w) // 2, (h - roi_h) // 2
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        
        # แปลง HSV และสร้าง mask
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_data = self.color_ranges[color_key]
        
        # สร้าง mask จากช่วงสี
        lower_bound = np.array(color_data[0])
        upper_bound = np.array(color_data[1])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # หา contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
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
                shape = 'rectangle'
            
            if shape == self.target_shape:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                center_x = roi_x + x + w_box // 2
                center_y = roi_y + y + h_box // 2
                results.append((center_x, center_y))
        
        return results
    
    def get_display_color(self):
        """ได้สีสำหรับการแสดงผลตามเป้าหมายปัจจุบัน"""
        if not self.target_color:
            return (255, 255, 255)  # สีขาวเป็นค่าเริ่มต้น
            
        # แปลงชื่อสีเก่าเป็นใหม่ (ถ้าจำเป็น)
        color_key = self.target_color
        if self.target_color in self.colors:
            color_key = self.colors[self.target_color]
        
        # คืนค่าสีสำหรับการแสดงผล
        if color_key in self.color_ranges:
            return self.color_ranges[color_key][2]  # index 2 คือสีสำหรับแสดงผล
        else:
            return (255, 255, 255)  # สีขาวเป็นค่าเริ่มต้น
    
    def get_available_colors(self):
        """ได้รายการสีที่ใช้งานได้"""
        legacy_colors = list(self.colors.keys())
        new_colors = list(self.color_ranges.keys())
        return {
            "legacy": legacy_colors,
            "new": new_colors,
            "all": legacy_colors + new_colors
        }


if __name__ == "__main__":
    # Example test code for detection using RoboMaster camera
    ep = robot.Robot()
    ep.initialize(conn_type="ap")  # เชื่อมต่อแบบ station mode (ปรับตามการเชื่อนต่อของคุณ)
    ep_camera = ep.camera
    ep_camera.start_video_stream(display=False)
    
    # ทดสอบ gimbal ก่อน
    print("🔧 ทดสอบระบบ gimbal...")
    gimbal_ok = test_gimbal_movement(ep)
    
    if not gimbal_ok:
        print("❌ gimbal ไม่ทำงาน - ตรวจสอบการเชื่อมต่อ")
        ep_camera.stop_video_stream()
        ep.close()
        exit(1)
    
    detector = CompactVisionDetector()
    
    # แสดงสีที่ใช้งานได้
    available_colors = detector.get_available_colors()
    print("🎨 สีที่ใช้งานได้:")
    print(f"  สีใหม่: {available_colors['new']}")
    print(f"  สีเก่า (legacy): {available_colors['legacy']}")
    
    # ตั้งค่าเป้าหมาย - ลองใช้สีใหม่
    target_color = "dark_green"  # เปลี่ยนได้: brown_red, dark_green, yellow_brown, dark_blue, dark_purple_red, vivid_dark_red
    target_shape = "circle"       # หรือ rectangle, triangle
    
    detector.set_target(target_color, target_shape)
    display_color = detector.get_display_color()

    print(f"🎥 เริ่มทดสอบการตรวจจับ - กด ESC เพื่อออก")
    print(f"🎯 กำลังหา: {target_color} {target_shape}")
    print(f"🌈 สีแสดงผล: {display_color}")
    
    try:
        while True:
            try:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=1.0)
                if frame is None:
                    print("⚠️ ไม่สามารถอ่านเฟรมได้")
                    continue

                detections = detector.detect(frame)
                
                # วาดผลการตรวจจับด้วยสีที่ถูกต้อง
                for (cx, cy) in detections:
                    cv2.circle(frame, (cx, cy), 20, display_color, 3)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # จุดแดงตรงกลาง
                    cv2.putText(frame, "TARGET", (cx-30, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)
                
                # แสดงข้อมูล
                cv2.putText(frame, f"Target: {target_color} {target_shape}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Found: {len(detections)} targets", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press ESC to quit, SPACE to shoot", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Compact Vision Detection Test", frame)
                
                # แสดงผลในคอนโซลด้วย
                if detections:
                    print(f"✅ พบเป้าหมาย {len(detections)} ตำแหน่ง: {detections}")
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC to quit
                    break
                elif key == 32:  # SPACE to shoot
                    if detections:
                        print("🎯 เริ่มการยิง!")
                        success = shoot_target_with_gimbal(ep, ep_camera, target_color, target_shape)
                        if success:
                            print("✅ ยิงสำเร็จ!")
                        else:
                            print("❌ ยิงไม่สำเร็จ!")
                    else:
                        print("⚠️ ไม่พบเป้าหมาย - ไม่สามารถยิงได้")
                    
            except Exception as e:
                print(f"❌ ข้อผิดพลาดในการอ่านเฟรม: {e}")
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n⚠️ ยกเลิกการทดสอบโดยผู้ใช้")
    finally:
        print("🔄 ปิดการเชื่อมต่อ...")
        ep_camera.stop_video_stream()
        ep.close()
        cv2.destroyAllWindows()
        print("✅ เสร็จสิ้น")