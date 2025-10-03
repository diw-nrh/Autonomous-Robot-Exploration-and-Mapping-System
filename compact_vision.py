"""
Compact Vision Detection for Project_2 with Object Tracking
=========================================================
ใช้สำหรับตรวจจับรูปทรงและสีในขั้นตอน scan หลังจากสแกนกำแพงเสร็จ
พร้อมฟีเจอร์หันตามวัตถุที่ detect ได้
"""

import cv2
import numpy as np
import time
import math
from robomaster import robot

class CompactVisionDetector:
    """ระบบตรวจจับรูปทรงและสีแบบย่อพร้อม object tracking"""
    
    def __init__(self, ep=None):
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
        self.ep = ep
        
        # การตั้งค่าสำหรับ object tracking
        self.frame_width = 640
        self.frame_height = 480
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        # Dead zone (พื้นที่ที่ไม่ต้องขยับ gimbal)
        self.dead_zone_x = 50  # ±50 pixels จากกลางแนวนอน
        self.dead_zone_y = 30  # ±30 pixels จากกลางแนวตั้ง
        
        # ความเร็ว gimbal (ปรับได้)
        self.gimbal_speed = 30  # degrees per second
        
    def set_target(self, color, shape):
        """ตั้งค่าเป้าหมาย"""
        self.target_color = color
        self.target_shape = shape
    
    def calculate_gimbal_movement(self, target_x, target_y):
        """คำนวณการเคลื่อนไหว gimbal ที่จำเป็น"""
        # คำนวณระยะห่างจากจุดกลาง
        dx = target_x - self.center_x
        dy = target_y - self.center_y
        
        # ตรวจสอบ dead zone
        if abs(dx) <= self.dead_zone_x and abs(dy) <= self.dead_zone_y:
            return None, None  # อยู่ใน dead zone ไม่ต้องขยับ
        
        # แปลง pixel เป็น degrees (ปรับค่าตามความต้องการ)
        # สมมติ FOV ประมาณ 60 degrees สำหรับ 640 pixels
        degrees_per_pixel_x = 60.0 / self.frame_width
        degrees_per_pixel_y = 45.0 / self.frame_height
        
        yaw_angle = -dx * degrees_per_pixel_x    # ลบเพราะ gimbal yaw ทำงานตรงกันข้าม
        pitch_angle = dy * degrees_per_pixel_y   # บวกเพราะต้องการให้ pitch ขึ้นลงตามปกติ
        
        # จำกัดมุมการเคลื่อนไหว
        yaw_angle = max(-10, min(10, yaw_angle))      # จำกัด ±10 degrees
        pitch_angle = max(-5, min(5, pitch_angle))    # จำกัด ±5 degrees
        
        return yaw_angle, pitch_angle
    
    def track_object(self, target_x, target_y):
        """หันหา gimbal ไปยังตำแหน่งวัตถุ"""
        if not self.ep:
            print("⚠️ ไม่มี robot instance สำหรับควบคุม gimbal")
            return False
        
        try:
            yaw_move, pitch_move = self.calculate_gimbal_movement(target_x, target_y)
            
            if yaw_move is None and pitch_move is None:
                # อยู่ใน dead zone แล้ว
                return True
            
            # ขยับ gimbal แบบ relative
            self.ep.gimbal.move(pitch=pitch_move, yaw=yaw_move, pitch_speed=self.gimbal_speed, yaw_speed=self.gimbal_speed)
            
            print(f"🎯 Tracking: target({target_x}, {target_y}) -> gimbal(yaw:{yaw_move:.1f}°, pitch:{pitch_move:.1f}°)")
            return True
            
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการควบคุม gimbal: {e}")
            return False
    
    def track_closest_object(self, detections):
        """หันหาวัตถุที่ใกล้จุดกลางที่สุด"""
        if not detections:
            return False
        
        # หาวัตถุที่ใกล้จุดกลางที่สุด
        closest_target = min(detections, key=lambda pos: 
                           math.sqrt((pos[0] - self.center_x)**2 + (pos[1] - self.center_y)**2))
        
        return self.track_object(closest_target[0], closest_target[1])
    
    def detect(self, frame):
        """ตรวจจับเป้าหมาย - คืนค่า [(center_x, center_y), ...] หรือ []"""
        if not self.target_color or not self.target_shape:
            return []
        
        # อัปเดตขนาดเฟรม
        self.frame_height, self.frame_width = frame.shape[:2]
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
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
    
    def detect_and_track(self, frame, auto_track=True):
        """ตรวจจับและหันตามวัตถุ (ถ้า auto_track=True)"""
        detections = self.detect(frame)
        
        if auto_track and detections:
            self.track_closest_object(detections)
        
        return detections
    
    def reset_gimbal(self):
        """รีเซ็ต gimbal กลับไปตำแหน่งกลาง"""
        if not self.ep:
            return False
        
        try:
            self.ep.gimbal.recenter().wait_for_completed()
            print("🔄 รีเซ็ต gimbal เรียบร้อย")
            return True
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการรีเซ็ต gimbal: {e}")
            return False
    
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


def test_gimbal_movement(ep):
    """ทดสอบการทำงานของ gimbal"""
    try:
        print("🔧 ทดสอบ gimbal...")
        ep.gimbal.recenter().wait_for_completed()
        time.sleep(0.5)
        ep.gimbal.move(yaw=10, pitch=0, yaw_speed=30).wait_for_completed()
        time.sleep(0.5)
        ep.gimbal.recenter().wait_for_completed()
        return True
    except Exception as e:
        print(f"❌ Gimbal ไม่ทำงาน: {e}")
        return False


if __name__ == "__main__":
    # Example test code for detection and tracking using RoboMaster camera
    ep = robot.Robot()
    ep.initialize(conn_type="ap")  # เชื่อมต่อแบบ station mode
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
    
    # สร้าง detector พร้อม robot instance
    detector = CompactVisionDetector(ep=ep)
    
    # แสดงสีที่ใช้งานได้
    available_colors = detector.get_available_colors()
    print("🎨 สีที่ใช้งานได้:")
    print(f"  สีใหม่: {available_colors['new']}")
    print(f"  สีเก่า (legacy): {available_colors['legacy']}")
    
    # ตั้งค่าเป้าหมาย - ลองใช้สีใหม่
    target_color = "blue"  # เปลี่ยนได้: brown_red, dark_green, yellow_brown, dark_blue, dark_purple_red, vivid_dark_red
    target_shape = "circle"       # หรือ rectangle, triangle
    
    detector.set_target(target_color, target_shape)
    display_color = detector.get_display_color()

    print(f"🎥 เริ่มทดสอบการตรวจจับและหันตาม - กด ESC เพื่อออก")
    print(f"🎯 กำลังหา: {target_color} {target_shape}")
    print(f"🌈 สีแสดงผล: {display_color}")
    print("📋 คำสั่ง:")
    print("  - SPACE: เปิด/ปิด auto tracking")
    print("  - R: รีเซ็ต gimbal")
    print("  - ESC: ออก")
    
    auto_tracking = True  # เปิด auto tracking ตั้งแต่เริ่มต้น
    
    try:
        while True:
            try:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=1.0)
                if frame is None:
                    print("⚠️ ไม่สามารถอ่านเฟรมได้")
                    continue

                # ตรวจจับและหันตาม (ถ้าเปิด auto tracking)
                detections = detector.detect_and_track(frame, auto_track=auto_tracking)
                
                # วาดผลการตรวจจับ
                for (cx, cy) in detections:
                    cv2.circle(frame, (cx, cy), 20, display_color, 3)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # จุดแดงตรงกลาง
                    cv2.putText(frame, "TARGET", (cx-30, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)
                
                # วาดจุดกลางและ dead zone
                cv2.circle(frame, (detector.center_x, detector.center_y), 5, (255, 255, 255), -1)
                cv2.rectangle(frame, 
                            (detector.center_x - detector.dead_zone_x, detector.center_y - detector.dead_zone_y),
                            (detector.center_x + detector.dead_zone_x, detector.center_y + detector.dead_zone_y),
                            (128, 128, 128), 1)
                
                # แสดงข้อมูล
                cv2.putText(frame, f"Target: {target_color} {target_shape}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Found: {len(detections)} targets", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Auto Tracking: {'ON' if auto_tracking else 'OFF'}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if auto_tracking else (0, 0, 255), 2)
                cv2.putText(frame, "ESC:quit | SPACE:toggle tracking | R:reset gimbal", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Compact Vision Detection with Tracking", frame)
                
                # แสดงผลในคอนโซลด้วย
                if detections:
                    closest = min(detections, key=lambda pos: 
                                math.sqrt((pos[0] - detector.center_x)**2 + (pos[1] - detector.center_y)**2))
                    print(f"✅ พบเป้าหมาย {len(detections)} ตำแหน่ง | ใกล้ที่สุด: {closest}")
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC to quit
                    break
                elif key == 32:  # SPACE to toggle auto tracking
                    auto_tracking = not auto_tracking
                    print(f"🔄 Auto tracking: {'เปิด' if auto_tracking else 'ปิด'}")
                elif key == ord('r') or key == ord('R'):  # R to reset gimbal
                    print("🔄 รีเซ็ต gimbal...")
                    detector.reset_gimbal()
                    
            except Exception as e:
                print(f"❌ ข้อผิดพลาดในการอ่านเฟรม: {e}")
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n⚠️ ยกเลิกการทดสอบโดยผู้ใช้")
    finally:
        print("🔄 รีเซ็ต gimbal และปิดการเชื่อมต่อ...")
        detector.reset_gimbal()
        ep_camera.stop_video_stream()
        ep.close()
        cv2.destroyAllWindows()
        print("✅ เสร็จสิ้น")