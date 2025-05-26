import cv2
from ultralytics import YOLO

# We will be using cvzone to display all the detections, as it makes it a little bit easier.
import cvzone
import math

# cap = cv2.VideoCapture(0) # For Webcam
# # # setting the width and height of the webcam
# cap.set(3, 1280) # width
# cap.set(4, 720) # height

cap = cv2.VideoCapture("../Videos/ppe-1-1.mp4") #  For Videos

# If you want to know what are the arguments a functions can take and want to have detailed information about it.
# Press Ctrl + Left Click on the function to understand it properly

model = YOLO('best.pt')

# Based on  Dataset which we have created using google colab
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
              'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van',
              'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']


while True:
    success, img = cap.read()
    results = model(img, stream = True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # -------------------- Displaying the Bounding Box
            x1, y1, x2, y2 = box.xyxy[0] # can also use box.xywh, to get x, y, width and height of the box
            # x1, y1, w, h = box.xywh[0]
            x1, y1, x2, y2 = int (x1), int (y1), int (x2), int(y2)

            # Using opencv, basic rectangles
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2-x1, y2-y1
            bbox = x1, y1, w, h

            conf = math.ceil((box.conf[0] * 100)) / 100

            # We will find the id number, use classNames[id] to get the object name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf > 0.5:
                if currentClass == 'Safety Vest' or currentClass == 'Mask' or currentClass == 'Hardhat' or currentClass == 'Gloves':
                    myColor = (0, 255, 0)
                elif currentClass == 'NO-Safety Vest' or currentClass == 'NO-Mask' or currentClass == 'NO-Hardhat':
                    myColor = (0, 0, 255)
                else:
                    myColor = (255, 0, 0)\

                cv2.rectangle(img, (x1, y1), (x2, y2), myColor)

                cvzone.putTextRect(img, f"{currentClass} {conf}",(max(0,x1), max(35, y1)),
                               scale = 1, thickness = 1, colorB = myColor, colorT = (255, 255, 255), colorR = myColor)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)