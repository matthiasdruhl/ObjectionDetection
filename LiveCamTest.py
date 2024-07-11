import yolov5
import cv2 
import numpy as np
import os
import DetectBalls
import time

# define a video capture object 
vid = cv2.VideoCapture(0) 

  
while(True): 
    
    
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    

    DetectBalls.find_ball(frame)

    # Display the resulting frame 
    #cv2.imwrite('video.jpg', frame) 

    
    time.sleep(10)

    

      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

    
  
# After the loop release the cap object 
# Destroy all the windows 
vid.release()
cv2.destroyAllWindows