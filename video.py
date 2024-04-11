
from imutils.video import FileVideoStream
import imutils
import time
import cv2

# Paths to Haar cascade XML files for face, eye, and smile detection
detector_paths = {
    "face": "cascades/haarcascade_frontalface_default.xml",
    "eyes": "cascades/haarcascade_eye.xml",
    "smile": "cascades/haarcascade_smile.xml"
}
# Load Haar cascade classifiers
detectors = {name: cv2.CascadeClassifier(path) for name, path in detector_paths.items()}

# Video file path
video_path = "noahfaces2.mp4"

vs = FileVideoStream(path=video_path).start()
time.sleep(1.0)

# Loop over frames
frame_number = 1
while True:
    # Grab the image from the video stream, resize it and then convert it to grayscale
    frame = vs.read()
    try:
        # Resize frame
        frame = imutils.resize(frame, width=500)
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detectors["face"].detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
        
        print("{} faces detected!".format(len(faces)))
            # Loop over detected faces
        for (x, y, w, h) in faces:
                # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Extract face region
            face_roi = gray[y:y+h, x:x+w]

                # Detect eyes
            eyes = detectors["eyes"].detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))

                # Loop over detected eyes
            for (ex, ey, ew, eh) in eyes:
                    # Draw rectangle around eye
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

                # Detect smiles
            smiles = detectors["smile"].detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))

                # Loop over detected smiles
            for (sx, sy, sw, sh) in smiles:
                    # Draw rectangle around smile
                cv2.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 0, 255), 2)

        # Show the output image frame
        cv2.imshow("Frame", frame)
        
        # Write frame to file
        image_path ="out/saved_image_"+ str(frame_number) +".jpg"
        frame_number = frame_number+1
        cv2.imwrite(image_path, frame)
            
        # Check for 'q' key press to exit loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("q pressed")
            break
    except:
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()