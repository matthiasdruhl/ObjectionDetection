

import numpy as np
import os
import time
from Motor import *
import RPi.GPIO as GPIO
import numpy





class Boundary_Detection:
    
    def __init__(self):
        self.IR01 = 14
        self.IR02 = 15
        self.IR03 = 23
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.IR01,GPIO.IN)
        GPIO.setup(self.IR02,GPIO.IN)
        GPIO.setup(self.IR03,GPIO.IN)
        self.order = ["Yellow"]
        self.gameOver = False
        self.locked = False
        self.LMR = 0
    
        
    
    def run(self):
        
        while True:
            
            self.LMR=0x00
            if GPIO.input(self.IR01)==True:
                self.LMR=(self.LMR | 4)
    
            if GPIO.input(self.IR02)==True:
                self.LMR=(self.LMR | 2)
    
            if GPIO.input(self.IR03)==True:
                self.LMR=(self.LMR | 1)
            continue
        
            
            
            
                
            if self.LMR == 2:
                PWM.setMotorModel(-500,-500,-500,-500)
            elif self.LMR==4:
                self.backup(.7)
                self.turnRight(.5)
      
            elif self.LMR==6:
                self.backup(.4)
                self.turnRight(1)
            elif self.LMR==1:
                self.backup(.7)
                self.turnLeft(.7)
            elif self.LMR==3:
                self.backup(.6)
                self.turnLeft(1)
            elif self.LMR==7 or self.LMR == 5:
                #pass
                self.backup(.9)
                self.turnRight(.9)
            else:
                continue
            if self.locked == True:
                if len(self.order) == 1:
                    self.gameOver = False
                else:
                    self.order = self.order[1:]
                    self.locked = False
            print(self.LMR)
            
          
            
    def stop(self, leng):
        PWM.setMotorModel(0,0,0,0)
        time.sleep(leng)
                
    def backup(self, leng):
        self.stop(.5)
        PWM.setMotorModel(500,500,500,500)
        time.sleep(leng)
        
    def turnLeft(self, leng):
        self.stop(.2)
        PWM.setMotorModel(1300, 1300, -1300, -1300)
        time.sleep(leng)
    
    def turnRight(self, leng):
        self.stop(.2)
        PWM.setMotorModel(-1300, -1300, 1300, 1300)
        time.sleep(leng)
        
           
  
    
        
        
boundary=Boundary_Detection()

# Main program logic follows:
if __name__ == '__main__':
    print ('Program is starting ... ')
    try:

        boundary.run()
    except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the child program  will be  executed.
        PWM.setMotorModel(0,0,0,0)
    
