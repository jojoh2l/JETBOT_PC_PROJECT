from jetbot import Robot

# สร้าง Instance ของหุ่นยนต์ครั้งเดียวเมื่อมีการ import ไฟล์นี้
robot = Robot()

def set_motors(left_pwm, right_pwm):
    """
    ฟังก์ชันสำหรับสั่งมอเตอร์
    - left_pwm: ค่าความเร็วล้อซ้าย (-1.0 ถึง 1.0)
    - right_pwm: ค่าความเร็วล้อขวา (-1.0 ถึง 1.0)
    """
    robot.left_motor.value = left_pwm
    robot.right_motor.value = right_pwm

def stop():
    """
    ฟังก์ชันสำหรับสั่งหยุดมอเตอร์ทันที
    """
    robot.stop()