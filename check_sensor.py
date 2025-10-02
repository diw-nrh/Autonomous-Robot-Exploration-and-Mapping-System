from robomaster import robot
def adc_to_distance(adc_value):
    return
# 0 = True (มีผนัง), 1 = False (ไม่มีผนัง)


if __name__ == "__main__":
    ep = robot.Robot()
    ep.initialize(conn_type="ap")
    ep_adaptor = ep.sensor_adaptor

    while True:
        sharp_left = not ep_adaptor.get_io(id=1, port=2) #for distance
        sharp_right = not ep_adaptor.get_adc(id=2, port=1) # for wall

        print("Sharp left : ", adc_to_distance(sharp_left), "Sharp right : ", adc_to_distance(sharp_right))