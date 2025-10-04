# -*- coding: utf-8 -*-
"""
Simple Gimbal Movement Library
=============================
ไลบรารีสำหรับการควบคุมการเคลื่อนไหว gimbal ของ RoboMaster แบบง่ายๆ
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
        >>> from lib.gimbal_simple import move_gimbal_to_center
        >>> ep = robot.Robot()
        >>> ep.initialize(conn_type="ap")
        >>> # สมมติเจอวัตถุที่ตำแหน่ง (800, 400)
        >>> success = move_gimbal_to_center(ep, 800, 400, debug=True)
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
    
    print("📖 Simple Gimbal Movement Functions Library")
    print("🔧 Available functions:")
    print("   - move_gimbal_to_center(ep_robot, center_x, center_y)")
    print("   - move_gimbal_smooth(ep_robot, center_x, center_y, duration)")
    print("   - recenter_gimbal(ep_robot)")
    print("   - reset_gimbal_pid()")
    print("")
    print("💡 Example:")
    print("   import robomaster.robot as robot")
    print("   from lib.gimbal_simple import move_gimbal_to_center")
    print("   ")
    print("   ep = robot.Robot()")
    print("   ep.initialize(conn_type='ap')")
    print("   # หัน gimbal ไปยังวัตถุที่ตำแหน่ง (800, 400)")
    print("   move_gimbal_to_center(ep, 800, 400, debug=True)")