from controller import Camera, Device, InertialUnit, Motor, GPS, PositionSensor, Robot, TouchSensor, Lidar
from controller import LidarPoint
import numpy as np
import math

TIME_STEP = 16

# PR2 constants
MAX_WHEEL_SPEED = 3.0  # maximum velocity for the wheels [rad / s]
WHEELS_DISTANCE = 0.4492  # distance between 2 caster wheels (the four wheels are located in square) [m]
SUB_WHEELS_DISTANCE = 0.098  # distance between 2 sub wheels of a caster wheel [m]
WHEEL_RADIUS = 0.08  # wheel radius

# function to check if a double is almost equal to another
TOLERANCE = 0.05  # arbitrary value

# ALMOST_EQUAL(a, b) = ((a < b + TOLERANCE) && (a > b - TOLERANCE))

# helper constants to distinguish the motors
FLL_WHEEL = 0
FLR_WHEEL = 1
FRL_WHEEL = 2
FRR_WHEEL = 3
BLL_WHEEL = 4
BLR_WHEEL = 5
BRL_WHEEL = 6
BRR_WHEEL = 7

FL_ROTATION = 0
FR_ROTATION = 1
BL_ROTATION = 2
BR_ROTATION = 3

SHOULDER_ROLL = 0
SHOULDER_LIFT = 1
UPPER_ARM_ROLL = 2
ELBOW_LIFT = 3
WRIST_ROLL = 4

LEFT_FINGER = 0
RIGHT_FINGER = 1
LEFT_TIP = 2
RIGHT_TIP = 3

# PR2 motors and their sensors
wheel_motors = []
wheel_sensors = []
rotation_motors = []
rotation_sensors = []
left_arm_motors = []
left_arm_sensors = []
right_arm_motors = []
right_arm_sensors = []
right_finger_motors = []
right_finger_sensors = []
left_finger_motors = []
left_finger_sensors = []
head_tilt_motor = []
torso_motor = []
torso_sensor = []

# PR2 sensor devices
left_finger_contact_sensors = []
right_finger_contact_sensors = []
imu_sensor = []
wide_stereo_l_stereo_camera_sensor = []
wide_stereo_r_stereo_camera_sensor = []
high_def_sensor = []
r_forearm_cam_sensor = []
l_forearm_cam_sensor = []
laser_tilt = []
base_laser = []
laser_tilt_width = 0
base_laser_width = 0
laser_tilt_maxRange = 0
base_laser_maxRange = 0
head_tilt_joint_sensor = []


def initialize_devices():
    wheel_motors.append(robot.getMotor("fl_caster_l_wheel_joint"))  # FLL_WHEEL
    wheel_motors.append(robot.getMotor("fl_caster_r_wheel_joint"))  # FLR_WHEEL
    wheel_motors.append(robot.getMotor("fr_caster_l_wheel_joint"))  # FRL_WHEEL
    wheel_motors.append(robot.getMotor("fr_caster_r_wheel_joint"))  # FRR_WHEEL
    wheel_motors.append(robot.getMotor("bl_caster_l_wheel_joint"))  # BLL_WHEEL
    wheel_motors.append(robot.getMotor("bl_caster_r_wheel_joint"))  # BLR_WHEEL
    wheel_motors.append(robot.getMotor("br_caster_l_wheel_joint"))  # BRL_WHEEL
    wheel_motors.append(robot.getMotor("br_caster_r_wheel_joint"))  # BRR_WHEEL
    for i in range(8):
        wheel_sensors.append(wheel_motors[i].getPositionSensor())

    rotation_motors.append(robot.getMotor("fl_caster_rotation_joint"))  # FL_ROTATION
    rotation_motors.append(robot.getMotor("fr_caster_rotation_joint"))  # FR_ROTATION
    rotation_motors.append(robot.getMotor("bl_caster_rotation_joint"))  # BL_ROTATION
    rotation_motors.append(robot.getMotor("br_caster_rotation_joint"))  # BR_ROTATION
    for i in range(4):
        rotation_sensors.append(rotation_motors[i].getPositionSensor())

    left_arm_motors.append(robot.getMotor("l_shoulder_pan_joint"))  # SHOULDER_ROLL
    left_arm_motors.append(robot.getMotor("l_shoulder_lift_joint"))  # SHOULDER_LIFT
    left_arm_motors.append(robot.getMotor("l_upper_arm_roll_joint"))  # UPPER_ARM_ROLL
    left_arm_motors.append(robot.getMotor("l_elbow_flex_joint"))  # ELBOW_LIFT
    left_arm_motors.append(robot.getMotor("l_wrist_roll_joint"))  # WRIST_ROLL
    for i in range(5):
        left_arm_sensors.append(left_arm_motors[i].getPositionSensor())

    right_arm_motors.append(robot.getMotor("r_shoulder_pan_joint"))  # SHOULDER_ROLL
    right_arm_motors.append(robot.getMotor("r_shoulder_lift_joint"))  # SHOULDER_LIFT
    right_arm_motors.append(robot.getMotor("r_upper_arm_roll_joint"))  # UPPER_ARM_ROLL
    right_arm_motors.append(robot.getMotor("r_elbow_flex_joint"))  # ELBOW_LIFT
    right_arm_motors.append(robot.getMotor("r_wrist_roll_joint"))  # WRIST_ROLL
    for i in range(5):
        right_arm_sensors.append(right_arm_motors[i].getPositionSensor())

    left_finger_motors.append(robot.getMotor("l_gripper_l_finger_joint"))  # LEFT_FINGER
    left_finger_motors.append(robot.getMotor("l_gripper_r_finger_joint"))  # RIGHT_FINGER
    left_finger_motors.append(robot.getMotor("l_gripper_l_finger_tip_joint"))  # LEFT_TIP
    left_finger_motors.append(robot.getMotor("l_gripper_r_finger_tip_joint"))  # RIGHT_TIP
    for i in range(4):
        left_finger_sensors.append(left_finger_motors[i].getPositionSensor())

    right_finger_motors.append(robot.getMotor("r_gripper_l_finger_joint"))  # LEFT_FINGER
    right_finger_motors.append(robot.getMotor("r_gripper_r_finger_joint"))  # RIGHT_FINGER
    right_finger_motors.append(robot.getMotor("r_gripper_l_finger_tip_joint"))  # LEFT_TIP
    right_finger_motors.append(robot.getMotor("r_gripper_r_finger_tip_joint"))  # RIGHT_TIP
    for i in range(4):
        right_finger_sensors.append(right_finger_motors[i].getPositionSensor())

    # head_tilt_joint_sensor.append(robot.getPositionSensor("head_tilt_joint_sensor"))
    # head_tilt_joint_sensor[0].enable(TIME_STEP)
    # head_tilt_motor.append(robot.getDevice("head_tilt_joint"))
    # torso_motor.append(robot.getMotor("torso_lift_joint"))
    # torso_sensor.append(robot.getMotor("torso_lift_joint_sensor"))

    # left_finger_contact_sensors.append(robot.getMotor("l_gripper_l_finger_tip_contact_sensor"))  # LEFT_FINGER
    # left_finger_contact_sensors.append(robot.getMotor("l_gripper_r_finger_tip_contact_sensor"))  # RIGHT_FINGER
    # right_finger_contact_sensors.append(robot.getMotor("r_gripper_l_finger_tip_contact_sensor"))  # LEFT_FINGER
    # right_finger_contact_sensors.append(robot.getMotor("r_gripper_r_finger_tip_contact_sensor"))  # RIGHT_FINGER
    # print(right_finger_contact_sensors[0])
    # # imu_sensor.append(Robot.getMotor("imu_sensor"))

    # laser_tilt.append(robot.getLidar("laser_tilt"))
    # base_laser.append(robot.getLidar("base_laser"))


# def enable_devices():
#     print("Enable device")
#     laser_tilt[0].enable(TIME_STEP)
#     base_laser[0].enable(TIME_STEP)
#     laser_tilt[0].enablePointCloud
#     base_laser[0].enablePointCloud
#     laser_tilt_width = laser_tilt[0].getHorizontalResolution
#     base_laser_width = base_laser[0].getHorizontalResolution
#     laser_tilt_maxRange = laser_tilt[0].getMaxRange
#     base_laser_maxRange = base_laser[0].getMaxRange
#     print(base_laser_maxRange)


# set the speeds of the robot wheels
def set_wheels_speeds(fll, flr, frl, frr, bll, blr, brl, brr):
    wheel_motors[FLL_WHEEL].setPosition(float('Inf'))
    wheel_motors[FLR_WHEEL].setPosition(float('Inf'))
    wheel_motors[FRL_WHEEL].setPosition(float('Inf'))
    wheel_motors[FRR_WHEEL].setPosition(float('Inf'))
    wheel_motors[BLL_WHEEL].setPosition(float('Inf'))
    wheel_motors[BLR_WHEEL].setPosition(float('Inf'))
    wheel_motors[BRL_WHEEL].setPosition(float('Inf'))
    wheel_motors[BRR_WHEEL].setPosition(float('Inf'))

    wheel_motors[FLL_WHEEL].setVelocity(fll)
    wheel_motors[FLR_WHEEL].setVelocity(flr)
    wheel_motors[FRL_WHEEL].setVelocity(frl)
    wheel_motors[FRR_WHEEL].setVelocity(frr)
    wheel_motors[BLL_WHEEL].setVelocity(bll)
    wheel_motors[BLR_WHEEL].setVelocity(blr)
    wheel_motors[BRL_WHEEL].setVelocity(brl)
    wheel_motors[BRR_WHEEL].setVelocity(brr)


def set_wheels_speed(speed):
    print("Set speed")
    set_wheels_speeds(speed, speed, speed, speed, speed, speed, speed, speed)


def stop_wheels():
    set_wheels_speeds(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def robot_go_forward(distance):
    print("Go forward")
    if distance > 0:
        max_wheel_speed = MAX_WHEEL_SPEED
    else:
        max_wheel_speed = -MAX_WHEEL_SPEED

    set_wheels_speed(max_wheel_speed)


# Idem for the right arm
def set_right_arm_position(shoulder_roll, shoulder_lift, upper_arm_roll, elbow_lift, wrist_roll):
    right_arm_motors[SHOULDER_ROLL].setPosition(shoulder_roll)
    right_arm_motors[SHOULDER_LIFT].setPosition(shoulder_lift)
    right_arm_motors[UPPER_ARM_ROLL].setPosition(upper_arm_roll)
    right_arm_motors[ELBOW_LIFT].setPosition(elbow_lift)
    right_arm_motors[WRIST_ROLL].setPosition(wrist_roll)


# Idem for the left arm
def set_left_arm_position(shoulder_roll, shoulder_lift, upper_arm_roll, elbow_lift, wrist_roll):
    left_arm_motors[SHOULDER_ROLL].setPosition(shoulder_roll)
    left_arm_motors[SHOULDER_LIFT].setPosition(shoulder_lift)
    left_arm_motors[UPPER_ARM_ROLL].setPosition(upper_arm_roll)
    left_arm_motors[ELBOW_LIFT].setPosition(elbow_lift)
    left_arm_motors[WRIST_ROLL].setPosition(wrist_roll)


def lidar_setting():
    temp = robot.getLidar("base_laser")
    temp.enable(TIME_STEP)

    base_laser_value = temp.getRangeImage()
    print("Lidar => Left:", base_laser_value[639], "Front:", base_laser_value[300], "Right:", base_laser_value[0])

    loc = gps.getValues()

    RFID = []
    for i in range(0, 640, 10):
        x = loc[0] + base_laser_value[i] * math.cos(-45 + (270 / 640) * i)
        y = loc[2] + base_laser_value[i] * math.sin(-45 + (270 / 640) * i)
        RFID.append([x, y])

    # print("RFID: ", RFID)
    return loc


def camara_setting():
    # print(camera.getWidth(), camera.getHeight())
    cameraData = camera.getImageArray()
    print(cameraData)


if __name__ == '__main__':
    robot = Robot()
    initialize_devices()

    # left_finger = robot.getMotor("l_gripper_l_finger_joint")
    # right_finger = robot.getMotor("l_gripper_r_finger_joint")
    slide = robot.getMotor("torso_lift_joint")
    slide.setPosition(0.1)
    left_finger_motors[LEFT_FINGER].setPosition(10)
    left_finger_motors[RIGHT_FINGER].setPosition(10)

    # camara_setting()
    camera = robot.getCamera("camera")
    camera.enable(TIME_STEP)
    camera.getWidth()
    camera.getHeight()

    gps = robot.getGPS("gps")
    gps.enable(TIME_STEP)
    imu = robot.getInertialUnit("imu_sensor")
    imu.enable(TIME_STEP)

    old_time = 0
    location = [0, 0, 0]
    old_angle = imu.getRollPitchYaw()[2]

    while robot.step(TIME_STEP * 10) != -1:
        # wheel_motors[FLL_WHEEL].setPosition(float('Inf'))
        # wheel_motors[FLR_WHEEL].setPosition(float('Inf'))
        # wheel_motors[FRL_WHEEL].setPosition(float('Inf'))
        # wheel_motors[FRR_WHEEL].setPosition(float('Inf'))
        # wheel_motors[BLL_WHEEL].setPosition(float('Inf'))
        # wheel_motors[BLR_WHEEL].setPosition(float('Inf'))
        # wheel_motors[BRL_WHEEL].setPosition(float('Inf'))
        # wheel_motors[BRR_WHEEL].setPosition(float('Inf'))
        #
        # wheel_motors[FLL_WHEEL].setVelocity(50)
        # wheel_motors[FLR_WHEEL].setVelocity(50)
        # wheel_motors[FRL_WHEEL].setVelocity(50)
        # wheel_motors[FRR_WHEEL].setVelocity(50)
        # wheel_motors[BLL_WHEEL].setVelocity(50)
        # wheel_motors[BLR_WHEEL].setVelocity(50)
        # wheel_motors[BRL_WHEEL].setVelocity(50)
        # wheel_motors[BRR_WHEEL].setVelocity(50)

        new_location = lidar_setting()
        print("new_location: ", new_location)
        new_time = robot.getTime()
        new_angle = imu.getRollPitchYaw()[2]

        velocity = ((new_location[0] - location[0]) ** 2 + (new_location[2] - location[2]) ** 2) ** 0.5 / (
                    new_time - old_time)
        if new_angle < 0:
            new_angle = new_angle + 2 * math.pi

        angle_velocity = (new_angle - old_angle) / (new_time - old_time)

        if abs(new_angle - old_angle) > 1 and new_angle > math.pi:
            angle_velocity = (new_angle - 2 * math.pi - old_angle) / (new_time - old_time)
        elif abs(new_angle - old_angle) > 1 and new_angle < math.pi:
            angle_velocity = (new_angle + 2 * math.pi - old_angle) / (new_time - old_time)
        else:
            angle_velocity = (new_angle - old_angle) / (new_time - old_time)

        print("Velocity: ", velocity)
        print("angle: ", old_angle)
        print("Angle velocity: ", angle_velocity)

        old_time = new_time
        location = new_location
        old_angle = new_angle

        cameraData = camera.getImageArray()
        # print(cameraData)
        print("=================================")


    # enable_devices()

    # set_left_arm_position(10, 10, 10, 10, 10)
    # set_right_arm_position(-10, 10, 10, 10, 10)
    #
    # set_left_arm_position(0.0, 1.35, 0.0, -2.2, 0.0)
    # set_right_arm_position(0.0, 1.35, 0.0, -2.2, 0.0)
