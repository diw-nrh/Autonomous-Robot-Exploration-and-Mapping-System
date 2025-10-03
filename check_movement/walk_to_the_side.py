
from robomaster import robot
import time


def short_left_right_loop():
    """สั้น ๆ: ขยับซ้ายแล้วขวาวนไปมา (small oscillation)
    กด Ctrl+C เพื่อหยุด
    """
    ep = robot.Robot()
    ep.initialize(conn_type="ap")
    chassis = ep.chassis

    try:
        while True:
            # ขยับไปซ้ายเล็กน้อย
            chassis.move(x=0, y=0.2, z=0, xy_speed=0.2).wait_for_completed()
            time.sleep(0.2)

            # ขยับไปขวาเล็กน้อย
            chassis.move(x=0, y=-0.2, z=0, xy_speed=0.2).wait_for_completed()
            time.sleep(0.2)

    except KeyboardInterrupt:
        # หยุดปลอดภัย
        chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        ep.close()


if __name__ == '__main__':
    print('Start small left-right oscillation. Ctrl+C to stop')
    short_left_right_loop()

