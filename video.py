from imutils.video import VideoStream
from imutils.video import FileVideoStream
import argparse
import imutils
import time
import cv2
import os

detectorPaths = {"face": "cascades/haarcascade_frontalface_default.xml",
                 "eyes": "cascades/haarcascade_eye.xml",
                 "smile": "cascades/haarcascade_smile.xml"}

# Load the Haar Cascade Face Detector from disk
print("Loading face detector")
detectors = {}

# Loop over the detector paths
for (name, path) in detectorPaths.items():
    # Load the Haar Cascade from disk and store it in the detectors dictionary
    detectors[name] = cv2.CascadeClassifier(path)

# Initialize the video stream and allow the camera sensor to warm up
print("Starting video stream")
video_path = "noahfaces2.mp4"
# vs = VideoStream(src=video_path ).start()
vs = FileVideoStream(path=video_path).start()
time.sleep(0.5)

print("start reading")
# Loop over the frames from the video stream
x = 1
while True:
    print("loop")
    # time.sleep(0.5)
    # Grab the image from the video stream, resize it and then convert it to grayscale
    frame = vs.read()
    # time.sleep(0.5)
    # vs.stream()
    try:
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Perform face detection
        print("Performing face detection")
        faceRects = detectors["face"].detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
        print("{} faces detected!".format(len(faceRects)))
        # Loop over the bounding boxes
        for (fX, fY, fW, fH) in faceRects:
        # Extract the face ROI
            faceROI = gray[fY:fY + fH, fX:fX + fW]
            # Apply eyes detection to the face ROI
            eyeRects = detectors["eyes"].detectMultiScale(faceROI, scaleFactor=1.1, minNeighbors=10,
                                                          minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
            # print("[INFO] {} eyes detected!".format(len(eyeRects)))
            # Apply smile detection to the face ROI
            smileRects = detectors["smile"].detectMultiScale(faceROI, scaleFactor=1.1, minNeighbors=10,
                                                             minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
            # print("[INFO] {} smiles detected!".format(len(smileRects)))
            # Loop over the eye bounding boxes
            for (eX, eY, eW, eH) in eyeRects:
                # Draw the eye bounding box
                ptA = (fX + eX, fY + eY)
                ptB = (fX + eX + eW, fY + eY + eH)
                cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)
            # Loop over the smile bounding boxes
            for (sX, sY, sW, sH) in smileRects:
                # Draw the smile bounding box
                ptA = (fX + sX, fY + sY)
                ptB = (fX + sX + sW, fY + sY + sH)
                cv2.rectangle(frame, ptA, ptB, (255, 0, 0), 2)
            # Draw the face bounding box on the frame
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

        # Show the output image frame
        cv2.imshow("Frame", frame)
        image_path ="out/saved_image_"+ str(x) +".jpg"
        x = x+1
        cv2.imwrite(image_path, frame)
        key = cv2.waitKey(1) & 0xFF
        # If the 'Q' key was pressed, break from the loop
        if key == ord("q"):
            print("q pressed")
            break
    except:
        print("no face")
        

print("done reading")
# Do a bit of cleanup
# cv2.destroyAllWindows()
vs.stop()