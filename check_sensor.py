from robomaster import robot
import time

def adc_to_distance_left(adc_value):
    """
    แปลงค่า ADC เป็น mm สำหรับ Sharp Left
    ข้อมูลการวัด:
    150mm -> 224 ADC
    100mm -> 357 ADC
    50mm  -> 698 ADC
    
    ใช้ linear interpolation และ extrapolation
    """
    if adc_value <= 0:
        return 999  # Invalid reading
    
    # ข้อมูลจุดอ้างอิง (adc, distance)
    points = [(224, 150), (357, 100), (698, 50)]
    
    # หาจุดที่ใกล้ที่สุด 2 จุดสำหรับ interpolation
    if adc_value <= 224:
        # Extrapolate จากจุดแรกสองจุด
        x1, y1 = 224, 150
        x2, y2 = 357, 100
    elif adc_value <= 357:
        # Interpolate ระหว่างจุดแรกและจุดที่สอง
        x1, y1 = 224, 150
        x2, y2 = 357, 100
    elif adc_value <= 698:
        # Interpolate ระหว่างจุดที่สองและจุดที่สาม
        x1, y1 = 357, 100
        x2, y2 = 698, 50
    else:
        # Extrapolate จากจุดสุดท้ายสองจุด
        x1, y1 = 357, 100
        x2, y2 = 698, 50
    
    # Linear interpolation: y = y1 + (y2-y1) * (x-x1) / (x2-x1)
    distance = y1 + (y2 - y1) * (adc_value - x1) / (x2 - x1)
    
    return max(20, min(300, distance))

def adc_to_distance_right(adc_value):
    """
    แปลงค่า ADC เป็น mm สำหรับ Sharp Right
    ข้อมูลการวัด:
    150mm -> 235 ADC
    100mm -> 364 ADC
    50mm  -> 705 ADC
    
    ใช้ linear interpolation และ extrapolation
    """
    if adc_value <= 0:
        return 999  # Invalid reading
    
    # ข้อมูลจุดอ้างอิง (adc, distance)
    points = [(235, 150), (364, 100), (705, 50)]
    
    # หาจุดที่ใกล้ที่สุด 2 จุดสำหรับ interpolation
    if adc_value <= 235:
        # Extrapolate จากจุดแรกสองจุด
        x1, y1 = 235, 150
        x2, y2 = 364, 100
    elif adc_value <= 364:
        # Interpolate ระหว่างจุดแรกและจุดที่สอง
        x1, y1 = 235, 150
        x2, y2 = 364, 100
    elif adc_value <= 705:
        # Interpolate ระหว่างจุดที่สองและจุดที่สาม
        x1, y1 = 364, 100
        x2, y2 = 705, 50
    else:
        # Extrapolate จากจุดสุดท้ายสองจุด
        x1, y1 = 364, 100
        x2, y2 = 705, 50
    
    # Linear interpolation: y = y1 + (y2-y1) * (x-x1) / (x2-x1)
    distance = y1 + (y2 - y1) * (adc_value - x1) / (x2 - x1)
    
    return max(20, min(300, distance))

def adc_to_distance(adc_value):
    """Generic function - ใช้สูตรเฉลี่ย"""
    return (adc_to_distance_left(adc_value) + adc_to_distance_right(adc_value)) / 2

# 0 = True (มีผนัง), 1 = False (ไม่มีผนัง)


if __name__ == "__main__":
    ep = robot.Robot()
    ep.initialize(conn_type="ap")
    ep_adaptor = ep.sensor_adaptor

    print("🔬 Sharp Sensor Distance Testing")
    print("=" * 40)
    print("ADC -> Distance conversion active")
    print("Press Ctrl+C to exit")
    print()

    try:
        while True:
            # อ่านค่า ADC
            sharp_left_adc = ep_adaptor.get_adc(id=1, port=2)   # Left sensor
            sharp_right_adc = ep_adaptor.get_adc(id=2, port=1)  # Right sensor

            # แปลงเป็น distance (mm)
            distance_left = adc_to_distance_left(sharp_left_adc)
            distance_right = adc_to_distance_right(sharp_right_adc)

            # แสดงผล
            print(f"Left  - ADC: {sharp_left_adc:4d} → {distance_left:6.1f} mm")
            print(f"Right - ADC: {sharp_right_adc:4d} → {distance_right:6.1f} mm")
            print("-" * 30)
            
            time.sleep(0.2)  # หน่วงเวลา 200ms

    except KeyboardInterrupt:
        print("\n🛑 Stopping sensor test...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        ep.close()
        print("✅ Robot disconnected safely")