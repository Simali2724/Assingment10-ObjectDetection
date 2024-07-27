import time

import cv2
import numpy as np

from sklearn.cluster import MeanShift


# Initialize start_time before the loop
start_time = time.time()


# Open the video filehgjhhg-09op4tyr45
cap = cv2.VideoCapture('video.mp4')
min_width_rect=80 #min
min_height_rect=80 #heightmin
count_line_position=550

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect = []
offset=6 #allowable error between pixel
counter=0

#initialize Substructor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video file.")
    exit()

# Initial green time (adjust based on your needs)```
green_time = 10  # seconds (example)
red_time = 5  # seconds (example)
current_state = "green"  # initial state

start_time = None  # time when green light starts


# Read and display frames from the video
while True:
    # Read a frame from the video
    ret, frame1 = cap.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey,(3,3),5)

   # applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,0),3)
    #MIN_SIZE = (30, 50)
    features = []
    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
         #features.append(extracted_feature)
       # if w > MIN_SIZE[0] and h > MIN_SIZE[1]:
        #    tracker = CamShift(frame, (x, y, w, h))
        validate_counter = (w>=min_width_rect) and (h>= min_height_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)

        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)

        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1
            cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)

            detect.remove((x,y))
            print("Vehicle:"+str(counter))

    cv2.putText(frame1,"VEHICLE:"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)

    if current_state == "green":
        if counter > 0:  # Calculate red light time only if vehicles have passed
            red_time: int = counter * 1.5  # Allocate 2 seconds per vehicle
            current_state = "red"
            start_time = time.time()
            counter = 0  # reset counter for next
            #-print(red_time)


    green_time_threshold = 10  # vehicles (example)  # Switch to red after this many vehicles pass

    # Update signal timer based on vehicle count or state
    if current_state == "green":
        if counter >= green_time_threshold:  # Switch to red after green time threshold is reached
            current_state = "red"
            start_time = time.time()
            counter = 0  # reset counter for next green phase
    elif current_state == "red":
        elapsed_time = time

    # Update signal timer based on state
    if current_state == "green":
        if start_time is not None:  # Check if start_time is assigned a value
            elapsed_time = time.time() - start_time
            if elapsed_time > green_time:  # Switch to red after green time
                current_state = "red"
                start_time = None
                counter = 0  # reset counter for next green phase
        else:
            start_time = time.time()  # Initialize start_time if not already set

            # Check if video finished or loop needs to terminate
            if not ret:
                # ... other code to handle end of video ...

                # Print timer if red light is ongoing
                if current_state == "red":
                   # elapsed_time = time.time() - start_time
                    elapsed_time = counter*2
                    print("Red light timer based on vehicle count:", elapsed_time, "seconds")

                break

    # Check elapsed time and stop after 1 minute
    elapsed_time = time.time() - start_time
    elapsed_time = counter * 1.5
    if elapsed_time > 40:  # 1 minute in seconds
        print("Vehicle counting stopped after 1 minute.")
        print("Greenxa  light timer based on vehicle count:", elapsed_time, "seconds")
        break


    # Check if the frame was read successfully
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Display the frame
    cv2.imshow('Video', frame1)

    # Check for key press to exit
    cv2.imshow('Detector', dilatada)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
