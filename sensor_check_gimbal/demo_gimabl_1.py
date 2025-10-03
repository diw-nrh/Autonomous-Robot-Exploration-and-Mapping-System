from robomaster import robot, camera
import cv2
import numpy as np
import time
import queue

# --- กำหนดช่วงสีและสีของกรอบ (HSV + BGR สำหรับ bounding box) ---
color_ranges = {
    "brown_red":      ([0, 150, 50],   [10, 255, 120],  (0, 0, 255)),
    "dark_green":     ([70, 150, 20],  [90, 255, 70],   (0, 255, 0)),
    "yellow_brown":   ([15, 150, 80],  [30, 255, 180],  (0, 255, 255)),
    "dark_blue":      ([110, 150, 30], [130, 255, 100], (255, 0, 0)),
    "dark_purple_red":([170, 150, 50], [180, 255, 150], (255, 0, 255)),
    "vivid_dark_red": ([170, 180, 50], [180, 255, 200], (255, 255, 255))
}

def detect_shape(contour):
    """ตรวจจับรูปทรงจาก contour"""
    # หา perimeter และ approximation
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    vertices = len(approx)
    
    # จำแนกรูปทรงตามจำนวนมุม
    if vertices >= 8:
        # ตรวจสอบว่าเป็นวงกลมหรือไม่
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.7:
            return 'circle'
        else:
            return 'polygon'
    elif vertices == 3:
        return 'triangle'
    elif vertices == 4:
        # ตรวจสอบว่าเป็นสี่เหลี่ยมจัตุรัสหรือสี่เหลี่ยมผืนผ้า
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.9 <= aspect_ratio <= 1.1:
            return 'square'
        else:
            return 'rectangle'
    else:
        return 'unknown'

def check_target_match(color_name, shape, target_color, target_shape):
    """ตรวจสอบว่าตรงกับเป้าหมายหรือไม่"""
    color_match = (target_color == "any" or color_name == target_color)
    shape_match = (target_shape == "any" or shape == target_shape)
    return color_match and shape_match

if __name__ == '__main__':
    # 🎯 กำหนดเป้าหมาย - แก้ไขค่าเหล่านี้ตามต้องการ
    TARGET_COLOR = "dark_green"  # หรือ "any" สำหรับทุกสี
    TARGET_SHAPE = "circle"      # หรือ "any" สำหรับทุกรูปทรง
    # ตัวเลือกสี: brown_red, dark_green, yellow_brown, dark_blue, dark_purple_red, vivid_dark_red, any
    # ตัวเลือกรูปทรง: circle, triangle, rectangle, square, polygon, any
    
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)

    print(f"🎯 กำลังค้นหา: {TARGET_COLOR} {TARGET_SHAPE}")
    print("📋 คำสั่ง:")
    print("  - กด 'q' เพื่อออก")
    print("  - กด '1-6' เพื่อเปลี่ยนสีเป้าหมาย")
    print("  - กด 'c/t/r/s' เพื่อเปลี่ยนรูปทรงเป้าหมาย")
    print("  - กด 'a' เพื่อตั้งเป้าหมายเป็น any (ทุกอย่าง)")

    # สร้าง mapping สำหรับคีย์
    color_keys = {
        '1': 'brown_red',
        '2': 'dark_green', 
        '3': 'yellow_brown',
        '4': 'dark_blue',
        '5': 'dark_purple_red',
        '6': 'vivid_dark_red'
    }
    
    shape_keys = {
        'c': 'circle',
        't': 'triangle',
        'r': 'rectangle', 
        's': 'square',
        'p': 'polygon'
    }
    
    target_found_count = 0

    try:
        while True:
            try:
                img = ep_camera.read_cv2_image(strategy="newest", timeout=1)
                if img is None:
                    continue
            except queue.Empty:
                continue

            frame = img.copy()
            frame_height, frame_width, _ = frame.shape

            # --- 1. กำหนดขอบเขตของ Region of Interest (ROI) ---
            roi_w = 800  # ความกว้างของ ROI
            roi_h = 450  # ความสูงของ ROI
            roi_x = (frame_width - roi_w) // 2
            roi_y = (frame_height - roi_h) // 2

            # --- 2. ตัดภาพ (Crop) ให้อยู่ในขอบเขต ROI เพื่อใช้ประมวลผล ---
            roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            # ประมวลผลภาพเฉพาะในส่วนของ ROI
            hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
            
            current_targets = []  # เก็บเป้าหมายที่พบในเฟรมนี้

            # 🔥 แก้ไขตรงนี้ - ประมวลผลเฉพาะสีที่ต้องการ
            if TARGET_COLOR == "any":
                # ถ้าต้องการทุกสี ให้วนลูปทุกสี
                colors_to_process = color_ranges.items()
            else:
                # ถ้าต้องการสีเฉพาะ ให้ประมวลผลเฉพาะสีนั้น
                if TARGET_COLOR in color_ranges:
                    colors_to_process = [(TARGET_COLOR, color_ranges[TARGET_COLOR])]
                else:
                    colors_to_process = []
                    print(f"⚠️ ไม่พบสี '{TARGET_COLOR}' ในระบบ")

            for color_name, (lower, upper, box_color) in colors_to_process:
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")
                mask = cv2.inRange(hsv, lower, upper)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 500:
                        # ตรวจจับรูปทรง
                        shape = detect_shape(cnt)
                        
                        # ตรวจสอบว่าตรงเป้าหมายหรือไม่
                        is_target = check_target_match(color_name, shape, TARGET_COLOR, TARGET_SHAPE)
                        
                        x, y, w, h = cv2.boundingRect(cnt)
                        draw_x = x + roi_x
                        draw_y = y + roi_y
                        
                        if is_target:
                            # เป้าหมายที่ต้องการ - ใช้กรอบหนาและสีพิเศษ
                            cv2.rectangle(frame, (draw_x, draw_y), (draw_x + w, draw_y + h), (0, 255, 0), 4)
                            cv2.putText(frame, f"TARGET: {color_name} {shape}", (draw_x, draw_y - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # วาดจุดกลาง
                            center_x = draw_x + w // 2
                            center_y = draw_y + h // 2
                            cv2.circle(frame, (center_x, center_y), 8, (0, 255, 0), -1)
                            
                            current_targets.append((color_name, shape, center_x, center_y))
                        else:
                            # วัตถุที่ไม่ตรงเป้าหมาย - แสดงเป็นสีเทาอ่อน
                            cv2.rectangle(frame, (draw_x, draw_y), (draw_x + w, draw_y + h), (128, 128, 128), 1)
                            cv2.putText(frame, f"{color_name} {shape}", (draw_x, draw_y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

            # นับเป้าหมายที่พบ
            if current_targets:
                target_found_count += 1
            
            # --- 4. วาดกรอบ ROI และข้อมูล ---
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 255), 2)
            
            # แสดงข้อมูลสถานะ
            cv2.putText(frame, f"Target: {TARGET_COLOR} {TARGET_SHAPE}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Found: {len(current_targets)} targets", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Total detections: {target_found_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # แสดงสีที่กำลังประมวลผล
            if TARGET_COLOR != "any":
                cv2.putText(frame, f"Processing: {TARGET_COLOR} only", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Processing: All colors", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # แสดงคำแนะนำ
            cv2.putText(frame, "1-6:color | c/t/r/s:shape | a:any | q:quit", (10, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            # แสดงผล
            cv2.imshow("Target Detection", frame)

            # จัดการคีย์
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif chr(key) in color_keys:
                TARGET_COLOR = color_keys[chr(key)]
                print(f"🎨 เปลี่ยนสีเป้าหมายเป็น: {TARGET_COLOR}")
            elif chr(key) in shape_keys:
                TARGET_SHAPE = shape_keys[chr(key)]
                print(f"🔷 เปลี่ยนรูปทรงเป้าหมายเป็น: {TARGET_SHAPE}")
            elif key == ord('a'):
                TARGET_COLOR = "any"
                TARGET_SHAPE = "any"
                print("🌈 ตั้งเป้าหมายเป็น: ทุกสีและทุกรูปทรง")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nโปรแกรมถูกหยุดโดยผู้ใช้ (Ctrl+C)")

    # แสดงสถิติสุดท้าย
    print(f"✅ สถิติการตรวจจับ:")
    print(f"   - เป้าหมาย: {TARGET_COLOR} {TARGET_SHAPE}")
    print(f"   - จำนวนครั้งที่พบ: {target_found_count}")

    # ปิดทุกอย่าง
    ep_camera.stop_video_stream()
    cv2.destroyAllWindows()
    ep_robot.close()