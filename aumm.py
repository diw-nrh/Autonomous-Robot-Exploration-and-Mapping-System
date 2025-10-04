from robomaster import robot, camera
import cv2
import numpy as np
import time
import queue
import math

# --- กำหนดช่วงสีและสีของกรอบ (HSV + BGR สำหรับ bounding box) ---

color_ranges = {
    'brown_red': ([0, 120, 70], [25, 255, 255],  (0, 0, 255)),
    'dark_green': ([50, 40, 40], [90, 255, 255],   (0, 255, 0)),
    'yellow_brown': ([10, 100, 100], [30, 255, 255],  (0, 255, 255)),
    'dark_blue': ([90, 60, 20], [150, 255, 255],(255, 0, 0))
}
# ---- ฟังก์ชันช่วย: จัดประเภท "รูปร่าง" ของวัตถุจากคอนทัวร์ ----
def classify_shape(cnt):
    """
    คืนค่า: 'circle' | 'square' | 'rectangle_h' | 'rectangle_v' | 'unknown'
    วิธีดู:
      - ใช้ circularity แยก 'circle'
      - ใช้ approxPolyDP ดูรูปสี่เหลี่ยม
      - ใช้อัตราส่วนกว้าง/สูง บอก square หรือ rectangle แนวนอน/แนวตั้ง
    """
    area = cv2.contourArea(cnt)
    if area < 1:
        return "unknown"

    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return "unknown"

    # 1) เช็ครูปทรงกลมด้วย "circularity"
    circularity = 4 * math.pi * (area / (peri * peri))
    if circularity > 0.75:
        return "circle"

    # 2) ใช้ approx ดูจำนวนมุม
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            return "unknown"
        ratio = w / float(h)
        # ขอบปลอดภัยนิด (allow 10%) สำหรับ square
        if 0.9 <= ratio <= 1.1:
            return "square"
        else:
            return "rectangle_h" if ratio > 1.1 else "rectangle_v"

    # อย่างอื่น ๆ ให้ unknown ไปก่อน (ถ้าต้องการ เพิ่ม triangle ฯลฯ ได้)
    return "unknown"

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_720P)

    print("กด q ที่หน้าต่างกล้อง หรือ Ctrl+C ใน terminal เพื่อหยุดโปรแกรม")

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

            # --- 1) กำหนด ROI ตรงกลางภาพ (ปรับได้) ---
            roi_w = 800
            roi_h = 450
            roi_x = (frame_width - roi_w) // 2
            roi_y = (frame_height - roi_h) // 2

            # --- 2) ประมวลผลเฉพาะ ROI ---
            roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

            # สำหรับดูรวม ๆ ถ้าต้องการ
            combined_mask = np.zeros(hsv.shape[:2], dtype="uint8")

            # เล็กน้อย: kernel สำหรับเปิด/ปิดรูปรอยรั่วของ mask
            kernel = np.ones((3, 3), np.uint8)

            for color_name, (lower, upper, box_color) in color_ranges.items():
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")
                mask = cv2.inRange(hsv, lower, upper)

                # ทำความสะอาด noise หน่อย
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

                combined_mask = cv2.bitwise_or(combined_mask, mask)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 500:
                        continue

                    x, y, w, h = cv2.boundingRect(cnt)

                    # --- 3) จัดประเภทวัตถุ (shape) ---
                    shape = classify_shape(cnt)

                    # --- 4) map พิกัดจาก ROI → ภาพเต็ม ---
                    draw_x = x + roi_x
                    draw_y = y + roi_y

                    # --- 5) วาดกรอบ + ใส่ label "สี_วัตถุ" ---
                    label = f"{color_name}_{shape}"
                    cv2.rectangle(frame, (draw_x, draw_y), (draw_x + w, draw_y + h), box_color, 2)

                    # จุดกึ่งกลาง (ช่วยเช็คการเล็ง)
                    cx = draw_x + w // 2
                    cy = draw_y + h // 2
                    cv2.circle(frame, (cx, cy), 3, box_color, -1)

                    # ใส่ชื่อวัตถุด้านบนกรอบ
                    cv2.putText(frame, label, (draw_x, max(0, draw_y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)

            # --- 6) วาดกรอบ ROI ให้เห็นขอบเขตบนภาพ ---
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 255), 2)

            # แสดงผล
            cv2.imshow("Frame", frame)
            cv2.imshow("Binary - Combined (ROI)", combined_mask)  # แสดงภาพไบนารีของ ROI

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.02)  # เว้นจังหวะเล็กน้อย

    except KeyboardInterrupt:
        print("\nโปรแกรมถูกหยุดโดยผู้ใช้ (Ctrl+C)")

    # ปิดทุกอย่าง
    ep_camera.stop_video_stream()
    cv2.destroyAllWindows()
    ep_robot.close()
