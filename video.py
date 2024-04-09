from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str, default="cascades",
                help="Path to haar cascade face detector .xml file")
args = vars(ap.parse_args())
detectorPaths = {"face": "haarcascade_frontalface_default.xml",
                 "eyes": "haarcascade_eye.xml",
                 "smile": "haarcascade_smile.xml"}

# Load the Haar Cascade Face Detector from disk
print("[INFO] Loading face detector...")
detectors = {}

# Loop over the detector paths
for (name, path) in detectorPaths.items():
    # Load the Haar Cascade from disk and store it in the detectors dictionary
    path = os.path.sep.join([args["cascade"], path])
    detectors[name] = cv2.CascadeClassifier(path)

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] Starting video stream...")
video_path = "noahfaces.mp4"
vs = VideoStream(src=video_path ).start()
time.sleep(1.0)

# Loop over the frames from the video stream
while True:
    # Grab the image from the video stream, resize it and then convert it to grayscale
    frame = vs.read()
    try:
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Perform face detection
        print("[INFO] Performing face detection...")
        faceRects = detectors["face"].detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
        print("[INFO] {} faces detected!".format(len(faceRects)))
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
        key = cv2.waitKey(1) & 0xFF
        # If the 'Q' key was pressed, break from the loop
        if key == ord("q"):
            break
    except:
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()