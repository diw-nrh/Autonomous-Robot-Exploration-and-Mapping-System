from robomaster import robot
import time

def adc_to_distance(adc_value):
    return

# 0 = True (มีผนัง), 1 = False (ไม่มีผนัง)

def setup_gimbal_follow_mode(ep_gimbal):
    """ตั้งค่า gimbal ให้หมุนตามรถ (follow mode)"""
    # Reset gimbal to center position
    ep_gimbal.recenter().wait_for_completed()
    
    # Enable gimbal follow mode - gimbal จะหมุนตามการเคลื่อนไหวของ chassis
    ep_gimbal.set_mode("follow")
    print("✅ Gimbal set to follow mode - จะหมุนตามรถ")

def manual_gimbal_control(ep_gimbal, yaw_angle=0):
    """ควบคุม gimbal แบบ manual"""
    ep_gimbal.moveto(yaw=yaw_angle, pitch=0, yaw_speed=90).wait_for_completed()
    print(f"🎯 Gimbal moved to yaw: {yaw_angle}°")

def gimbal_track_movement(ep_gimbal, ep_chassis, direction="left"):
    """ทำให้ gimbal หมุนไปทางทิศทางที่รถกำลังจะเคลื่อนไหว"""
    if direction == "left":
        ep_gimbal.moveto(yaw=-90, pitch=0, yaw_speed=180).wait_for_completed()
        print("🔄 Gimbal looking left")
    elif direction == "right":
        ep_gimbal.moveto(yaw=90, pitch=0, yaw_speed=180).wait_for_completed()
        print("🔄 Gimbal looking right")
    elif direction == "forward":
        ep_gimbal.moveto(yaw=0, pitch=0, yaw_speed=180).wait_for_completed()
        print("🔄 Gimbal looking forward")
    elif direction == "backward":
        ep_gimbal.moveto(yaw=180, pitch=0, yaw_speed=180).wait_for_completed()
        print("🔄 Gimbal looking backward")
    
    
if __name__ == "__main__":
    ep = robot.Robot()
    ep.initialize(conn_type="ap")
    
    # Initialize components
    ep_adaptor = ep.sensor_adaptor
    ep_chassis = ep.chassis
    ep_gimbal = ep.gimbal
    
    print("🤖 Robot initialized")
    print("📡 Setting up gimbal...")
    
    # วิธีที่ 1: ใช้ Follow Mode (แนะนำ)
    setup_gimbal_follow_mode(ep_gimbal)
    
    # วิธีที่ 2: Manual control (ถ้าต้องการควบคุมเอง)
    # manual_gimbal_control(ep_gimbal, yaw_angle=0)
    
    print("🚗 Starting movement with gimbal tracking...")
    
    # ทดสอบการเคลื่อนไหวกับ gimbal
    try:
        # หมุนขวา 90 องศา
        print("\n🔄 Turning right...")
        gimbal_track_movement(ep_gimbal, ep_chassis, "right")
        ep_chassis.move(x=0, y=0, z=90, xy_speed=0.3).wait_for_completed()
        time.sleep(1)
        
        # กลับมามองหน้า
        gimbal_track_movement(ep_gimbal, ep_chassis, "forward")
        time.sleep(1)
        
        print("\n📊 Starting sensor monitoring...")
        
        # Main sensor loop
        while True:
            ir_left_back = not ep_adaptor.get_io(id=1, port=1) #for distance
            ir_left_center = not ep_adaptor.get_io(id=1, port=2) # for wall
            ir_left_front = not ep_adaptor.get_io(id=2, port=2) #for distance 
            ir_right_center = not ep_adaptor.get_io(id=3, port=1) # for wall
            sharp_right_front = ep_adaptor.get_adc(id=3, port=2) #for distance
            sharp_right_back = ep_adaptor.get_adc(id=4, port=1) #for distance

            print("IR left : ", ir_left_front,ir_left_center, ir_left_back, "IR and sharp right : ", ir_right_center, sharp_right_front, sharp_right_back)
            
            # Example: Auto gimbal movement based on sensor data
            if not ir_left_center:  # มีกำแพงด้านซ้าย
                gimbal_track_movement(ep_gimbal, ep_chassis, "left")
                time.sleep(0.5)
            elif not ir_right_center:  # มีกำแพงด้านขวา
                gimbal_track_movement(ep_gimbal, ep_chassis, "right")
                time.sleep(0.5)
            else:  # ไม่มีกำแพง มองหน้า
                gimbal_track_movement(ep_gimbal, ep_chassis, "forward")
            
            time.sleep(0.1)  # Small delay
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping robot...")
        ep_gimbal.recenter().wait_for_completed()
        ep.close()