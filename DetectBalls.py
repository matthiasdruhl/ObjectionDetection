import yolov5
import cv2
import numpy as np
import os
import time
from sklearn.cluster import KMeans

global cType
from CameraType import CameraType
cType = CameraType()

def get_dominant_colors(pixels, k=3):
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Retrieve the dominant colors
    colors = kmeans.cluster_centers_

    # Convert the pixel values to integer
    return colors.astype(int)


def preprocess_image(image):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Convert to floating point
    pixels = np.float32(pixels)

    return pixels

def get_color(imageFrame):

    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
     
  
    
    preprocessed_image = preprocess_image(hsvFrame)

    num_colors = 1
    dominant_colors = (get_dominant_colors(pixels=preprocessed_image, k=num_colors))
    dominant_colors = dominant_colors.tolist()
    dominant_colors = dominant_colors[0]
    

    dom0 = dominant_colors[0]
    #dom1 = dominant_colors[1]
    
    #dom2 = dominant_colors[2]

    print(dom0)

    if dom0 in range(106, 180): #and dom1 in range(52,256) and dom2 in range(111, 255):
        print("RED")
    elif dom0 in range(40, 80): #and dom1 in range(52, 255) and dom2 in range(72, 255):
        print("GREEN")  
    elif dom0 in range(81, 105): #and dom1 in range(80, 255) and dom2 in range(2, 255):
        print("BLUE")  
    else:
        print("YELLOW")


def find_ball(img):
    #cType.setType("balls")
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    

    model_name = 'Yolov5_models'
    yolov5_model = 'balls5n.pt'
    model_labels = 'balls5n.txt'

    CWD_PATH = os.getcwd()
    PATH_TO_LABELS = os.path.join(CWD_PATH,model_name,model_labels)
    PATH_TO_YOLOV5_GRAPH = os.path.join(CWD_PATH,model_name,yolov5_model)

    # Import Labels File
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Initialize Yolov5
    model = yolov5.load(PATH_TO_YOLOV5_GRAPH)

    min_conf_threshold = 0.7 # I changed this
    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = True # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

    frame = img.copy()
    results = model(frame)
    predictions = results.pred[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    classes = predictions[:, 5]
    # Draws Bounding Box onto image
    #results.render() 
    

    # Initialize frame rate calculation
    frame_rate_calc = 30
    freq = cv2.getTickFrequency()

    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    #imW, imH = int(400), int(300)
    imW, imH = int(640), int(640)
    frame_resized = cv2.resize(frame_rgb, (imW, imH))
    #input_data = np.expand_dims(frame_resized, axis=0)

    max_score = 0
    #max_index = 0
    
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        curr_score = scores.numpy()

        # Found desired object with decent confidence
        if ((labels[int(classes[i])] == cType.getType()) and (curr_score[i] > max_score) and (curr_score[i] > min_conf_threshold) and (curr_score[i] <= 1.0)):
            time.sleep(3)
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            xmin = int(max(1,(boxes[i][0])))
            ymin = int(max(1,(boxes[i][1])))
            xmax = int(min(imW,(boxes[i][2])))
            ymax = int(min(imH,(boxes[i][3])))
                       
            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(curr_score[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            #cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            #if cType.getType() == "ball":
                
            # Record current max
            max_score = curr_score[i]
            #max_index = i
            cropped = frame[ymin:ymax, xmin: xmax]
            get_color(cropped)
            

    # Write Image (with bounding box) to file
    cv2.imwrite('video.jpg', frame)

    #cv2.imshow('video.jpg', frame)
#if __name__ == '__main__':
    
#    img = cv2.imread('image.jpg')
#    find_ball(img)