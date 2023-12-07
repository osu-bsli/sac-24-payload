import time
import board
from adafruit_motorkit import MotorKit
print("creating object")
kit = MotorKit(i2c = board.I2C())
tr = True
kit.motor2.throttle = 0
kit.motor3.throttle = 0
while tr:
	kit.motor2.throttle = 1.0
	kit.motor3.throttle = 1.0
	time.sleep(0.5)
	kit.motor2.throttle = 0
	kit.motor3.throttle = 0
	time.sleep(0.5)
