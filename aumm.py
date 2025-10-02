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

            # --- 1. กำหนดขอบเขตของ Region of Interest (ROI) ---
            # คุณสามารถปรับค่าเหล่านี้ได้ตามต้องการ
            roi_w = 800  # ความกว้างของ ROI
            roi_h = 450  # ความสูงของ ROI
            # คำนวณตำแหน่ง x, y เพื่อให้ ROI อยู่ตรงกลางภาพ
            roi_x = (frame_width - roi_w) // 2
            roi_y = (frame_height - roi_h) // 2

            # --- 2. ตัดภาพ (Crop) ให้อยู่ในขอบเขต ROI เพื่อใช้ประมวลผล ---
            roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            # ประมวลผลภาพเฉพาะในส่วนของ ROI
            hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
            combined_mask = np.zeros(hsv.shape[:2], dtype="uint8")

            for color_name, (lower, upper, box_color) in color_ranges.items():
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")
                mask = cv2.inRange(hsv, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 500:
                        x, y, w, h = cv2.boundingRect(cnt)

                        # --- 3. คำนวณตำแหน่งสำหรับวาดบนภาพเต็ม (สำคัญ!) ---
                        # บวกค่า x, y ของ ROI กลับเข้าไป เพื่อให้ตำแหน่งถูกต้องบน frame หลัก
                        draw_x = x + roi_x
                        draw_y = y + roi_y

                        cv2.rectangle(frame, (draw_x, draw_y), (draw_x + w, draw_y + h), box_color, 2)
                        cv2.putText(frame, color_name, (draw_x, draw_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # --- 4. วาดกรอบ ROI เพื่อให้เห็นขอบเขตบนภาพที่แสดงผล ---
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 255), 2) # สีเหลือง

            # แสดงผล
            cv2.imshow("Frame", frame)
            # cv2.imshow("Mask", combined_mask) # สามารถเปิดดู mask ได้หากต้องการ

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nโปรแกรมถูกหยุดโดยผู้ใช้ (Ctrl+C)")

    # ปิดทุกอย่าง
    ep_camera.stop_video_stream()
    cv2.destroyAllWindows()
    ep_robot.close()
