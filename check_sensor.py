from robomaster import robot
def adc_to_distance(adc_value):
    return
# 0 = Trye (มีผนัง), 1 = False (ไม่มีผนัง)
    
    
if __name__ == "__main__":
    ep = robot.Robot()
    ep.initialize(conn_type="ap")
    ep_adaptor = ep.sensor_adaptor
    while True:
        ir_left_back = not ep_adaptor.get_io(id=1, port=1) #for distance
        ir_left_center = not ep_adaptor.get_io(id=1, port=2) # for wall
        ir_left_front = not ep_adaptor.get_io(id=2, port=2) #for distance 
        ir_right_center = not ep_adaptor.get_io(id=3, port=1) # for wall
        sharp_right_front = ep_adaptor.get_adc(id=3, port=2) #for distance
        sharp_right_back = ep_adaptor.get_adc(id=4, port=1) #for distance

        
        print("IR left : ", ir_left_front,ir_left_center, ir_left_back, "IR and sharp right : ", ir_right_center, sharp_right_front, sharp_right_back)