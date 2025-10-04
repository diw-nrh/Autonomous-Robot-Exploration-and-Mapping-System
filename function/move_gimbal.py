# -*- coding: utf-8 -*-
"""
Gimbal Movement Functions Library
===============================
ไลบรารีสำหรับการควบคุมการเคลื่อนไหว gimbal ของ RoboMaster
เรียกใช้ฟังก์ชันเดียวเพื่อหัน gimbal ไปยังตำแหน่งกลางของวัตถุ
"""

import time
import math

# ========= Camera/Gimbal Constants =========
CAMERA_W = 1280
CAMERA_H = 720
CAMERA_HFOV_DEG = 120.0
ASPECT = CAMERA_W / CAMERA_H
CAMERA_VFOV_DEG = math.degrees(2.0 * math.atan(math.tan(math.radians(CAMERA_HFOV_DEG/2.0)) / ASPECT))
GIMBAL_TURN_SPEED = 360

# ========= PID Controller =========
class PID:
    """PID Controller สำหรับการควบคุม gimbal"""
    
    def __init__(self, kp=0.2, ki=0.0, kd=0.0, integ_clip=40.0, out_clip=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integ = 0.0
        self.prev_err = 0.0
        self.integ_clip = float(integ_clip)
        self.out_clip = float(out_clip)
    
    def reset(self):
        """รีเซ็ต PID state"""
        self.integ = 0.0
        self.prev_err = 0.0
    
    def step(self, err, dt):
        """คำนวณ PID output"""
        dt = max(dt, 1e-3)
        self.integ += err * dt
        self.integ = max(-self.integ_clip, min(self.integ, self.integ_clip))
        deriv = (err - self.prev_err) / dt
        self.prev_err = err
        out = self.kp * err + self.ki * self.integ + self.kd * deriv
        return max(-self.out_clip, min(self.out_clip, out))

# ========= Global PID instances =========
_pid_yaw = PID(kp=0.3, ki=0.0, kd=0.0, integ_clip=40.0, out_clip=15.0)
_pid_pitch = PID(kp=0.3, ki=0.0, kd=0.0, integ_clip=40.0, out_clip=12.0)
_last_time = time.monotonic()

def move_gimbal_to_center(ep_robot, center_x, center_y, frame_width=CAMERA_W, frame_height=CAMERA_H, 
                         move_threshold=0.1, max_retries=3, debug=False):
    """
    หัน gimbal ไปยังตำแหน่งกลางของวัตถุ - ฟังก์ชันเดียวทำครบทุกอย่าง
    
    Args:
        ep_robot: RoboMaster robot instance (ต้องเชื่อมต่อแล้ว)
        center_x (int): ตำแหน่ง x ของจุดกลางวัตถุใน frame (pixels)
        center_y (int): ตำแหน่ง y ของจุดกลางวัตถุใน frame (pixels)
        frame_width (int): ความกว้างของ frame (default: 1280)
        frame_height (int): ความสูงของ frame (default: 720)
        move_threshold (float): เกณฑ์ขั้นต่ำสำหรับการเคลื่อนไหว (degrees)
        max_retries (int): จำนวนครั้งที่พยายามเคลื่อนไหว
        debug (bool): แสดงข้อมูล debug หรือไม่
    
    Returns:
        bool: True ถ้าหัน gimbal สำเร็จ, False ถ้าล้มเหลว
    
    Example:
        >>> import robomaster.robot as robot
        >>> ep = robot.Robot()
        >>> ep.initialize(conn_type="ap")
        >>> # สมมติเจอวัตถุที่ตำแหน่ง (800, 400)
        >>> success = move_gimbal_to_center(ep, 800, 400)
        >>> if success:
        >>>     print("หัน gimbal สำเร็จ!")
    """
    global _last_time
    
    try:
        # คำนวณเวลา
        now = time.monotonic()
        dt = now - _last_time
        _last_time = now
        
        # คำนวณจุดกลางของ frame
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0
        
        # คำนวณ error ในหน่วย pixels
        px_off_x = center_x - frame_center_x
        px_off_y = center_y - frame_center_y
        
        # แปลงเป็น normalized error (-1.0 ถึง 1.0)
        err_norm_x = px_off_x / frame_center_x
        err_norm_y = px_off_y / frame_center_y
        
        # แปลงเป็น degrees
        yaw_err_deg = err_norm_x * (CAMERA_HFOV_DEG / 2.0)
        pitch_err_deg = -err_norm_y * (CAMERA_VFOV_DEG / 2.0)  # ลบเพื่อให้ทิศทางถูกต้อง
        
        # คำนวณ PID output
        yaw_cmd = _pid_yaw.step(yaw_err_deg, dt)
        pitch_cmd = _pid_pitch.step(pitch_err_deg, dt)
        
        if debug:
            print(f"🎯 Target: ({center_x}, {center_y}) | Frame center: ({frame_center_x}, {frame_center_y})")
            print(f"📊 Error: yaw={yaw_err_deg:.2f}°, pitch={pitch_err_deg:.2f}°")
            print(f"🔧 PID Command: yaw={yaw_cmd:.2f}°, pitch={pitch_cmd:.2f}°")
        
        # เคลื่อนไหว gimbal ถ้า command มีค่าเพียงพอ
        if abs(yaw_cmd) > move_threshold or abs(pitch_cmd) > move_threshold:
            for attempt in range(max_retries):
                try:
                    ep_robot.gimbal.move(
                        yaw=yaw_cmd, 
                        pitch=pitch_cmd,
                        yaw_speed=GIMBAL_TURN_SPEED, 
                        pitch_speed=GIMBAL_TURN_SPEED
                    ).wait_for_completed()
                    
                    if debug:
                        print(f"✅ Gimbal moved successfully (attempt {attempt + 1})")
                    
                    return True
                    
                except Exception as e:
                    if debug:
                        print(f"⚠️ Gimbal move attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(0.1)  # รอก่อนลองใหม่
        else:
            if debug:
                print("🎯 Target already near center (within threshold)")
            return True
            
    except Exception as e:
        if debug:
            print(f"❌ Error in move_gimbal_to_center: {e}")
        return False

def move_gimbal_smooth(ep_robot, center_x, center_y, duration=1.0, frame_width=CAMERA_W, frame_height=CAMERA_H, debug=False):
    """
    หัน gimbal ไปยังตำแหน่งแบบนุ่มนวลในระยะเวลาที่กำหนด
    
    Args:
        ep_robot: RoboMaster robot instance
        center_x (int): ตำแหน่ง x ของจุดกลางวัตถุ
        center_y (int): ตำแหน่ง y ของจุดกลางวัตถุ  
        duration (float): ระยะเวลาในการเคลื่อนไหว (วินาที)
        frame_width (int): ความกว้างของ frame
        frame_height (int): ความสูงของ frame
        debug (bool): แสดงข้อมูล debug
    
    Returns:
        bool: True ถ้าสำเร็จ, False ถ้าล้มเหลว
    """
    start_time = time.monotonic()
    success_count = 0
    total_attempts = 0
    
    if debug:
        print(f"🎯 Starting smooth gimbal movement to ({center_x}, {center_y}) for {duration:.1f}s")
    
    while (time.monotonic() - start_time) < duration:
        success = move_gimbal_to_center(
            ep_robot, center_x, center_y, 
            frame_width, frame_height, 
            move_threshold=0.05, debug=False
        )
        
        if success:
            success_count += 1
        total_attempts += 1
        
        time.sleep(0.05)  # 20 Hz update rate
    
    success_rate = success_count / total_attempts if total_attempts > 0 else 0
    
    if debug:
        print(f"✅ Smooth movement completed: {success_count}/{total_attempts} successful ({success_rate*100:.1f}%)")
    
    return success_rate > 0.8  # ถือว่าสำเร็จถ้าทำสำเร็จมากกว่า 80%

def reset_gimbal_pid():
    """รีเซ็ต PID controllers"""
    global _last_time
    _pid_yaw.reset()
    _pid_pitch.reset()
    _last_time = time.monotonic()
    print("🔄 PID controllers reset")

def recenter_gimbal(ep_robot, reset_pid=True):
    """
    รีเซ็ต gimbal กลับไปตำแหน่งกลาง
    
    Args:
        ep_robot: RoboMaster robot instance
        reset_pid (bool): รีเซ็ต PID controllers ด้วยหรือไม่
    
    Returns:
        bool: True ถ้าสำเร็จ, False ถ้าล้มเหลว
    """
    try:
        print("🔄 Recentering gimbal...")
        ep_robot.gimbal.recenter().wait_for_completed()
        
        if reset_pid:
            reset_gimbal_pid()
        
        print("✅ Gimbal recentered successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to recenter gimbal: {e}")
        return False

# ========= Example Usage =========
if __name__ == "__main__":
    """
    ตัวอย่างการใช้งาน:
    
    1. ฟังก์ชันหลัก - หัน gimbal ครั้งเดียว:
       move_gimbal_to_center(ep_robot, center_x, center_y)
    
    2. ฟังก์ชันแบบนุ่มนวล - หัน gimbal แบบต่อเนื่อง:
       move_gimbal_smooth(ep_robot, center_x, center_y, duration=2.0)
    
    3. รีเซ็ต gimbal:
       recenter_gimbal(ep_robot)
    """
    
    print("📖 Gimbal Movement Functions Library")
    print("🔧 Available functions:")
    print("   - move_gimbal_to_center(ep_robot, center_x, center_y)")
    print("   - move_gimbal_smooth(ep_robot, center_x, center_y, duration)")
    print("   - recenter_gimbal(ep_robot)")
    print("   - reset_gimbal_pid()")
    print("")
    print("💡 Example:")
    print("   import robomaster.robot as robot")
    print("   from lib.move_gimbal import move_gimbal_to_center")
    print("   ")
    print("   ep = robot.Robot()")
    print("   ep.initialize(conn_type='ap')")
    print("   # หัน gimbal ไปยังวัตถุที่ตำแหน่ง (800, 400)")
    print("   move_gimbal_to_center(ep, 800, 400, debug=True)")

# ========= Gimbal Movement Functions =========

class GimbalController:
    """คลาสสำหรับควบคุม gimbal ด้วย PID"""
    
    def __init__(self, ep_robot):
        self.ep_robot = ep_robot
        self.ep_gimbal = ep_robot.gimbal
        
        # PID Controllers
        self.pid_yaw = PID(kp=0.2, ki=0.0, kd=0.0, integ_clip=40.0, out_clip=10.0)
        self.pid_pitch = PID(kp=0.2, ki=0.0, kd=0.0, integ_clip=40.0, out_clip=8.0)
        
        # Gimbal state tracking
        self.gimbal_state = {"yaw": 0.0, "pitch": 0.0}
        self.last_time = time.monotonic()
        
        # Camera settings
        self.frame_width = CAMERA_W
        self.frame_height = CAMERA_H
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        self._setup_gimbal()
    
    def _setup_gimbal(self):
        """ตั้งค่า gimbal และ callback"""
        def gimbal_callback(angle_info):
            pitch_angle, yaw_angle, pitch_ground_angle, yaw_ground_angle = angle_info
            self.gimbal_state["yaw"] = float(yaw_angle)
            self.gimbal_state["pitch"] = float(pitch_angle)
        
        try:
            # Recenter gimbal
            print("🎯 กำลัง recenter gimbal...")
            self.ep_gimbal.recenter().wait_for_completed()
            time.sleep(0.5)
            
            # Start angle subscription
            self.ep_gimbal.sub_angle(freq=20, callback=gimbal_callback)
            
            # Wait for first callback
            t_wait = time.monotonic()
            while (time.monotonic() - t_wait) < 1.0:
                time.sleep(0.01)
                if self.gimbal_state["yaw"] != 0.0 or self.gimbal_state["pitch"] != 0.0:
                    break
            
            print("✅ ตั้งค่า gimbal เรียบร้อย")
            
        except Exception as e:
            print(f"⚠️ ข้อผิดพลาดในการตั้งค่า gimbal: {e}")
    
    def move_to_target(self, target_x, target_y, move_threshold=0.05):
        """
        เคลื่อนที่ gimbal ไปยังตำแหน่งเป้าหมาย
        
        Args:
            target_x (int): ตำแหน่ง x ของเป้าหมายใน frame (pixels)
            target_y (int): ตำแหน่ง y ของเป้าหมายใน frame (pixels) 
            move_threshold (float): เกณฑ์ขั้นต่ำสำหรับการเคลื่อนไหว (degrees)
        
        Returns:
            dict: ข้อมูลการเคลื่อนไหว {yaw_cmd, pitch_cmd, yaw_err, pitch_err, moved}
        """
        # คำนวณเวลา
        now = time.monotonic()
        dt = now - self.last_time
        self.last_time = now
        
        # คำนวณ error ในหน่วย pixels
        half_w = self.frame_width / 2.0
        half_h = self.frame_height / 2.0
        
        px_off_x = target_x - half_w
        px_off_y = target_y - half_h
        
        # แปลงเป็น normalized error (-1.0 ถึง 1.0)
        err_norm_x = px_off_x / half_w
        err_norm_y = px_off_y / half_h
        
        # แปลงเป็น degrees
        yaw_err_deg = err_norm_x * (CAMERA_HFOV_DEG / 2.0)
        pitch_err_deg = -err_norm_y * (CAMERA_VFOV_DEG / 2.0)  # ลบเพื่อให้ทิศทางถูกต้อง
        
        # คำนวณ PID output
        yaw_cmd = self.pid_yaw.step(yaw_err_deg, dt)
        pitch_cmd = self.pid_pitch.step(pitch_err_deg, dt)
        
        # เคลื่อนไหว gimbal ถ้า command มีค่าเพียงพอ
        moved = False
        if abs(yaw_cmd) > move_threshold or abs(pitch_cmd) > move_threshold:
            try:
                self.ep_gimbal.move(
                    yaw=yaw_cmd, 
                    pitch=pitch_cmd,
                    yaw_speed=GIMBAL_TURN_SPEED, 
                    pitch_speed=GIMBAL_TURN_SPEED
                )
                moved = True
            except Exception as e:
                print(f"⚠️ ข้อผิดพลาดในการเคลื่อนไหว gimbal: {e}")
        
        return {
            'yaw_cmd': yaw_cmd,
            'pitch_cmd': pitch_cmd,
            'yaw_err': yaw_err_deg,
            'pitch_err': pitch_err_deg,
            'moved': moved,
            'target_x': target_x,
            'target_y': target_y,
            'center_x': half_w,
            'center_y': half_h
        }
    
    def track_target_smooth(self, target_x, target_y, duration=1.0, fps=30):
        """
        หันตาม target แบบนุ่มนวลในระยะเวลาที่กำหนด
        
        Args:
            target_x (int): ตำแหน่ง x ของเป้าหมาย
            target_y (int): ตำแหน่ง y ของเป้าหมาย
            duration (float): ระยะเวลาในการ track (วินาที)
            fps (int): ความถี่ในการอัปเดต (Hz)
        
        Returns:
            list: รายการข้อมูลการเคลื่อนไหวในแต่ละ frame
        """
        results = []
        start_time = time.monotonic()
        frame_interval = 1.0 / fps
        
        print(f"🎯 เริ่ม tracking เป้าหมาย ({target_x}, {target_y}) เป็นเวลา {duration:.1f} วินาที")
        
        while (time.monotonic() - start_time) < duration:
            frame_start = time.monotonic()
            
            # เคลื่อนไหว gimbal
            result = self.move_to_target(target_x, target_y)
            results.append(result)
            
            # แสดงข้อมูล debug
            if result['moved']:
                print(f"   📊 Error: yaw={result['yaw_err']:.2f}°, pitch={result['pitch_err']:.2f}° | "
                      f"Cmd: yaw={result['yaw_cmd']:.2f}°, pitch={result['pitch_cmd']:.2f}°")
            
            # รอให้ครบ frame interval
            elapsed = time.monotonic() - frame_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
        
        print(f"✅ tracking เสร็จสิ้น ({len(results)} frames)")
        return results
    
    def reset_pid(self):
        """รีเซ็ต PID controllers"""
        self.pid_yaw.reset()
        self.pid_pitch.reset()
        print("🔄 รีเซ็ต PID controllers")
    
    def recenter(self):
        """รีเซ็ต gimbal กลับไปตำแหน่งกลาง"""
        try:
            print("🔄 กำลัง recenter gimbal...")
            self.ep_gimbal.recenter().wait_for_completed()
            self.reset_pid()
            print("✅ recenter gimbal เรียบร้อย")
            return True
        except Exception as e:
            print(f"❌ ข้อผิดพลาดในการ recenter: {e}")
            return False
    
    def cleanup(self):
        """ปิดการ subscription"""
        try:
            self.ep_gimbal.unsub_angle()
        except Exception:
            pass

# ========= Standalone Functions =========

def move_gimbal_to_position(ep_robot, target_x, target_y, move_threshold=0.05):
    """
    ฟังก์ชันแบบง่ายสำหรับการเคลื่อนที่ gimbal ไปยังตำแหน่งที่กำหนด
    
    Args:
        ep_robot: RoboMaster robot instance
        target_x (int): ตำแหน่ง x ของเป้าหมายใน frame (pixels)
        target_y (int): ตำแหน่ง y ของเป้าหมายใน frame (pixels)
        move_threshold (float): เกณฑ์ขั้นต่ำสำหรับการเคลื่อนไหว (degrees)
    
    Returns:
        bool: True ถ้าเคลื่อนไหวสำเร็จ, False ถ้าล้มเหลว
    """
    controller = GimbalController(ep_robot)
    try:
        result = controller.move_to_target(target_x, target_y, move_threshold)
        return result['moved']
    except Exception as e:
        print(f"❌ ข้อผิดพลาดในการเคลื่อนไหว gimbal: {e}")
        return False
    finally:
        controller.cleanup()

def track_object_center(ep_robot, center_x, center_y, duration=2.0):
    """
    หันตาม center ของวัตถุแบบต่อเนื่อง
    
    Args:
        ep_robot: RoboMaster robot instance
        center_x (int): ตำแหน่ง x ของจุดกลางวัตถุ
        center_y (int): ตำแหน่ง y ของจุดกลางวัตถุ
        duration (float): ระยะเวลาในการ track (วินาที)
    
    Returns:
        list: รายการข้อมูลการเคลื่อนไหว
    """
    controller = GimbalController(ep_robot)
    try:
        return controller.track_target_smooth(center_x, center_y, duration)
    except Exception as e:
        print(f"❌ ข้อผิดพลาดในการ track วัตถุ: {e}")
        return []
    finally:
        controller.cleanup()

# ========= utils =========
def _clip(v, lo, hi): return max(lo, min(int(v), hi))

def _tight_bbox_from_mask(mask_bool):
    ys, xs = np.where(mask_bool)
    if ys.size == 0: return None
    y0, y1 = int(ys.min()), int(ys.max())+1
    x0, x1 = int(xs.min()), int(xs.max())+1
    return x0, y0, x1, y1  # x_min, y_min, x_max(excl), y_max(excl)

def _tpl_to_bin_u8(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    return (v > 100).astype(np.uint8) * 255

def detect_and_measure(img_bgr):
    """
    return:
      vis_bgr, center_xy or None, conf(float), size_x(int), size_y(int), dist_x_cm(float), dist_y_cm(float), dist_avg_cm(float)
    """
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (h > HSV_H_LO) & (h < HSV_H_HI) & (s > HSV_S_MIN) & (v > HSV_V_MIN)
    mask_u8 = (mask.astype(np.uint8) * 255)

    # โหลด template
    tpl_bins = []
    for p in TEMPLATE_PATHS:
        t = cv2.imread(p)
        if t is None:
            raise FileNotFoundError(f"ไม่พบไฟล์เทมเพลต: {p}")
        tpl_bins.append(_tpl_to_bin_u8(t))

    # ตรวจสอบขนาด
    for i, T in enumerate(tpl_bins):
        th, tw = T.shape[:2]
        if th > H or tw > W:
            raise ValueError(f"Template index {i} ใหญ่กว่าภาพ (th={th}, tw={tw}, H={H}, W={W})")

    # template matching
    scores = []
    locs = []
    sizes = []
    for T in tpl_bins:
        val = cv2.matchTemplate(mask_u8, T, cv2.TM_CCOEFF_NORMED)
        _, m, _, loc = cv2.minMaxLoc(val)
        scores.append(m); locs.append(loc); sizes.append(T.shape[:2])

    best_idx = int(np.argmax(scores))
    max_val = float(scores[best_idx])
    top_left = locs[best_idx]
    th, tw = sizes[best_idx]

    vis = img_bgr.copy()
    cx = cy = np.nan
    size_x = size_y = 0
    dist_x = dist_y = dist_avg = 0.0
    center = None

    if max_val > TEMPLATE_MATCH_THRESHOLD:
        # window จากตำแหน่งแมตช์
        x0, y0 = int(top_left[0]), int(top_left[1])
        x1, y1 = x0 + tw, y0 + th
        x0c, y0c = _clip(x0, 0, W-1), _clip(y0, 0, H-1)
        x1c, y1c = _clip(x1, 1, W), _clip(y1, 1, H)

        win_mask = mask[y0c:y1c, x0c:x1c]
        tight = _tight_bbox_from_mask(win_mask)
        if tight is not None:
            tx0, ty0, tx1, ty1 = tight
            X0, Y0 = x0c + tx0, y0c + ty0
            X1, Y1 = x0c + tx1, y0c + ty1
            cv2.rectangle(vis, (X0, Y0), (X1, Y1), (0, 255, 0), 2)

            size_x = int(X1 - X0)
            size_y = int(Y1 - Y0)

            if size_x > 0: dist_x = KPIX_X * OBJ_W_CM / float(size_x)
            if size_y > 0: dist_y = KPIX_Y * OBJ_H_CM / float(size_y)
            # ถ้ามีทั้งสองแกน ใช้เฉลี่ย, ถ้ามีแกนเดียว ใช้อันนั้น
            if dist_x > 0 and dist_y > 0:
                dist_avg = 0.5 * (dist_x + dist_y)
            else:
                dist_avg = max(dist_x, dist_y)

            cx = int((X0 + X1) / 2)
            cy = int((Y0 + Y1) / 2)
            center = (cx, cy)

            # วาด crosshair/ข้อความ
            cv2.circle(vis, (cx, cy), 4, (0, 255, 255), -1)
            cv2.circle(vis, (W//2, H//2), 4, (255, 0, 0), -1)
            cv2.line(vis, (W//2, 0), (W//2, H), (255, 0, 0), 1)
            cv2.line(vis, (0, H//2), (W, H//2), (255, 0, 0), 1)
            cv2.putText(vis, f"conf={max_val:.2f}  w={size_x}px h={size_y}px",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.putText(vis, f"dist_x={dist_x:.2f}cm  dist_y={dist_y:.2f}cm  avg={dist_avg:.2f}cm",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        else:
            max_val = 0.0  # หน้าต่างไม่มีพิกเซลผ่าน mask
    else:
        # ไม่ผ่านเกณฑ์
        pass

    return vis, center, max_val, size_x, size_y, dist_x, dist_y, dist_avg

def safe_flush(file_obj, path):
    try:
        file_obj.flush(); os.fsync(file_obj.fileno())
    except Exception as e:
        print("Flush/Fsync error:", repr(e), file=sys.stderr)

def abs_path(p):
    try: return os.path.abspath(p)
    except Exception: return p

if __name__ == "__main__":
    print("Working directory:", abs_path(os.getcwd()))
    print(f"HFOV={CAMERA_HFOV_DEG:.2f}°, VFOV={CAMERA_VFOV_DEG:.2f}°")

    ep_robot = None; ep_camera = None; ep_gimbal = None

    try:
        ep_robot = robot.Robot(); ep_robot.initialize(conn_type="ap")
        ep_gimbal = ep_robot.gimbal
        ep_camera = ep_robot.camera

        try: ep_gimbal.recenter().wait_for_completed()
        except Exception: pass

        # ---- seed/subscribe มุมกิมบอล (กันค่าว่าง) ----
        gimbal_state = {"yaw": 0.0, "pitch": 0.0}
        _got = {"ok": False}
        def _gimbal_cb(ang):
            pitch_angle, yaw_angle, pitch_ground_angle, yaw_ground_angle = ang
            gimbal_state["yaw"]   = float(yaw_angle)
            gimbal_state["pitch"] = float(pitch_angle)
            _got["ok"] = True
        ep_gimbal.sub_angle(freq=20, callback=_gimbal_cb)

        t_wait0 = time.monotonic()
        while not _got["ok"] and (time.monotonic()-t_wait0) < 1.0:
            time.sleep(0.01)

        est_yaw = gimbal_state["yaw"]
        est_pitch = gimbal_state["pitch"]

        ep_camera.start_video_stream(display=False, resolution="720p")
        print("เริ่มสตรีม — กด Q เพื่อออก")

        pid_yaw   = PID(kp=0.20, ki=0.00, kd=0.00, integ_clip=40.0, out_clip=10.0)
        pid_pitch = PID(kp=0.20, ki=0.00, kd=0.00, integ_clip=40.0, out_clip= 8.0)

        t0 = time.monotonic(); last_t = t0

        while True:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=2)
            now = time.monotonic(); dt = now - last_t; last_t = now
            if frame is None:
                time.sleep(0.01); continue

            vis, center, conf, sx, sy, dx, dy, davg = detect_and_measure(frame)

            yaw_err_deg = pitch_err_deg = 0.0
            yaw_cmd = pitch_cmd = 0.0
            cx = cy = ""

            if center is not None:
                cx_i, cy_i = center
                cx, cy = int(cx_i), int(cy_i)

                # error เป็นองศาตามสัดส่วนพิกเซล
                half_w = CAMERA_W / 2.0
                half_h = CAMERA_H / 2.0
                err_x = (cx - half_w) / half_w
                err_y = (cy - half_h) / half_h

                yaw_err_deg   =  err_x * (CAMERA_HFOV_DEG/2.0)
                pitch_err_deg = -err_y * (CAMERA_VFOV_DEG/2.0)

                yaw_cmd   = pid_yaw.step(yaw_err_deg, dt)
                pitch_cmd = pid_pitch.step(pitch_err_deg, dt)

                if abs(yaw_cmd) > 0.05 or abs(pitch_cmd) > 0.05:
                    try:
                        ep_gimbal.move(yaw=yaw_cmd, pitch=pitch_cmd,
                                       yaw_speed=GIMBAL_TURN_SPEED, pitch_speed=GIMBAL_TURN_SPEED
                                       ).wait_for_completed()
                    except Exception:
                        try:
                            ep_gimbal.move(yaw=yaw_cmd, pitch=pitch_cmd,
                                           yaw_speed=GIMBAL_TURN_SPEED, pitch_speed=GIMBAL_TURN_SPEED)
                        except Exception:
                            pass
            else:
                # soften integral ถ้าไม่เห็นเป้า
                pid_yaw.integ *= 0.9
                pid_pitch.integ *= 0.9

            # อัปเดต estimator ให้ไม่ว่าง
            est_yaw  += yaw_cmd
            est_pitch += pitch_cmd
            yaw_out   = gimbal_state["yaw"]   if isinstance(gimbal_state["yaw"], (int,float)) else est_yaw
            pitch_out = gimbal_state["pitch"] if isinstance(gimbal_state["pitch"], (int,float)) else est_pitch

            # overlay เพิ่ม
            cv2.putText(vis, f"dist_avg={davg:.2f} cm", (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)
            cv2.putText(vis, f"yaw_err={yaw_err_deg:+.2f} pitch_err={pitch_err_deg:+.2f}",
                        (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)
            cv2.putText(vis, f"pid_yaw={yaw_cmd:+.2f} pid_pitch={pitch_cmd:+.2f}",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)

            cv2.imshow("Measure + PID track (yellow object)", vis)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt — stopping.")
    except Exception:
        print("\n[ERROR] Unhandled exception:", file=sys.stderr)
        traceback.print_exc()
    finally:
        try:
            if ep_camera is not None: ep_camera.stop_video_stream()
        except Exception: pass
        try:
            if ep_gimbal is not None: ep_gimbal.unsub_angle()
        except Exception: pass
        try:
            if ep_robot is not None: ep_robot.close()
        except Exception: pass
        cv2.destroyAllWindows()
