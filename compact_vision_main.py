"""
Compact Vision Detection for Project_2 with Object Tracking
=========================================================
ใช้สำหรับตรวจจับรูปทรงและสีในขั้นตอน scan หลังจากสแกนกำแพงเสร็จ
พร้อมฟังก์ชัน gimbal tracking และการยิง
"""

import cv2
import numpy as np
import time
import math
import queue
from robomaster import robot

class CompactVisionDetector:
    """ระบบตรวจจับรูปทรงและสีแบบย่อพร้อม object tracking"""
    
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

def run_robomaster_detection(target_color='red', target_shape='circle'):
    """
    รันการตรวจจับวัตถุด้วยกล้อง RoboMaster แบบสั้นๆ
    
    target_color: สีเป้าหมาย ('red', 'green', 'blue', 'yellow')
    target_shape: รูปทรงเป้าหมาย
        - 'circle': วงกลม
        - 'square': สี่เหลี่ยมจัตุรัส  
        - 'vertical_rectangle': สี่เหลี่ยมผืนผ้าแนวตั้ง
        - 'horizontal_rectangle': สี่เหลี่ยมผืนผ้าแนวนอน
    """
    import robomaster.robot as robot
    import cv2
    import queue
    
    print(f"🎯 เริ่มตรวจจับ {target_color} {target_shape}")
    print("📖 รูปทรงที่รองรับ:")
    print("   - circle: วงกลม")
    print("   - square: สี่เหลี่ยมจัตุรัส")
    print("   - vertical_rectangle: สี่เหลี่ยมผืนผ้าแนวตั้ง")
    print("   - horizontal_rectangle: สี่เหลี่ยมผืนผ้าแนวนอน")
    print("กด 'q' เพื่อหยุดโปรแกรม")
    
    try:
        # เชื่อมต่อ RoboMaster
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="ap")
        ep_camera = ep_robot.camera
        ep_camera.start_video_stream(display=False, resolution=robot.camera.STREAM_720P)
        
        # ตั้งค่า detector
        detector = CompactVisionDetector()
        detector.set_target(target_color, target_shape)
        
        while True:
            try:
                img = ep_camera.read_cv2_image(strategy="newest", timeout=1)
                if img is None:
                    continue
            except queue.Empty:
                continue
            
            # ตรวจจับวัตถุ
            detections = detector.detect(img)
            
            # วาดกรอบรอบวัตถุที่พบ
            for (x, y) in detections:
                cv2.rectangle(img, (x-30, y-30), (x+30, y+30), (0, 255, 0), 2)
                cv2.putText(img, f"{target_shape}", (x-40, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # แสดงข้อมูล
            cv2.putText(img, f"Detecting: {target_color} {target_shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"Found: {len(detections)} objects", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "Press 'q' to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("RoboMaster Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"❌ ข้อผิดพลาด: {e}")
    finally:
        try:
            ep_camera.stop_video_stream()
            cv2.destroyAllWindows()
            ep_robot.close()
        except:
            pass

if __name__ == "__main__":
    # ตัวอย่างการใช้งาน - เปลี่ยนรูปทรงได้ 4 แบบ
    print("🎯 ตัวอย่างการใช้งานรูปทรงต่างๆ:")
    print("1. วงกลมสีแดง: run_robomaster_detection('red', 'circle')")
    print("2. สี่เหลี่ยมจัตุรัสสีเขียว: run_robomaster_detection('green', 'square')")
    print("3. สี่เหลี่ยมผืนผ้าแนวตั้งสีน้ำเงิน: run_robomaster_detection('blue', 'vertical_rectangle')")
    print("4. สี่เหลี่ยมผืนผ้าแนวนอนสีเหลือง: run_robomaster_detection('yellow', 'horizontal_rectangle')")
    
    # รันตัวอย่างพื้นฐาน - วงกลมสีแดง
    run_robomaster_detection('blue', 'square')