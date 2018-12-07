import cv2
import math
import numpy as np
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
import time
import atexit

mh = Adafruit_MotorHAT(addr=0x60) # 0x60 is the default addr.
    
def turnOffMotors():
    # recommended for auto-disabling motors on shutdown!
    mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)

def forwards():
    leftMotor = mh.getMotor(3)
    rightMotor = mh.getMotor(4)

    # Set speed from 0 (off) to 255 (full speed) #
    leftMotor.setSpeed(150); 
    rightMotor.setSpeed(150);

    # Set motor direction #
    leftMotor.run(Adafruit_MotorHAT.FORWARD)
    rightMotor.run(Adafruit_MotorHAT.FORWARD)
    time.sleep(1)
    turnOffMotors()


def adjustHeading(delta):
    leftMotor = mh.getMotor(3)
    rightMotor = mh.getMotor(4)

    # Set speed from 0 (off) to 255 (full speed) #
    leftMotor.setSpeed(150); 
    rightMotor.setSpeed(150);
    
    if delta < 0:
        rightMotor.run(Adafruit_MotorHAT.FORWARD)
    else:
        leftMotor.run(Adafruit_MotorHAT.FORWARD)
    time.sleep(abs(delta))


def main():
    # Create Objects #
    cam = cv2.VideoCapture(0)
    atexit.register(turnOffMotors) # causes motors to stop on python code exit

    retval, frame1 = cam.read()
    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) # Convert frame to Grayscale
    print("Beginning Obstacal Avoidance program...")

    cnt = 1
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    while(1):
        # move forwards a fixed small distance #
        forwards()

        retval, frame2 = cam.read()
        filename = "frame_"+str(cnt)+".jpg"
        cnt = cnt+1;
        cv2.imwrite(filename,frame2)
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) # Convert frame to Grayscale
        
        # Caculate Optical Flow Vectors # 
        flow = cv2.calcOpticalFlowFarneback(prev, next, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx = flow[:,:,0] # first channel
        dy = flow[:,:,1] # second channel

        # Save Depth image
        mag, ang = cv2.cartToPolar(flow[...,0],flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        filename = "depth_"+str(cnt)+".jpg"
        cv2.imwrite(filename, bgr)


        # Get sum of motion vectors in each half of the image #
        left_sum = 0
        right_sum = 0
        
        Width = math.floor(.5 * np.shape(dx)[0])
        Width = int(Width)

        # Left Half #
        for i in range(0, Width-1):
            for j in range(0, np.shape(dx)[1]):
                left_sum = left_sum + math.sqrt(dx[i,j]**2 + dy[i,j]**2)
        
        # Right Half #
        for i in range(Width, np.shape(dx)[0]):
            for j in range(0, np.shape(dx)[1]):
                right_sum = right_sum + math.sqrt(dx[i,j]**2 + dy[i,j]**2)
        
        #print(left_sum)
        #print(right_sum)
        balance = (left_sum - right_sum) / (left_sum + right_sum)
        print(balance)
        adjustHeading(balance)

        # update for next cycle
        prev = next

if __name__ == '__main__':
    main()


