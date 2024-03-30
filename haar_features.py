import cv2
import matplotlib.pyplot as plt

# largely from https://www.datacamp.com/tutorial/face-detection-python-opencv
# using haar cascades accessed at https://github.com/opencv/opencv/tree/master/data/haarcascades


# cropping with the face values instead of selecting
# retrieves a smaller image with cut out backgrounds
def cropFace(imagePath):
    img = cv2.imread(imagePath)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # use opencv haar cascade
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
    )

    # (x,y) and width/height can be used to determine the coordinates to crop at
    for (x, y, w, h) in face:
        roi = img[y:y+h, x:x+w]

    return roi



# repurposing cropFace above and drawing lines to identify eyes and mouth
def facialFeatures(imagePath):
    img = cropFace(imagePath)
    #img = cv2.imread(imagePath)
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # use opencv haar cascade
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    eyes_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    mouth_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml"
    )

    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
    )

    eyes = eyes_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
    )

    mouth = mouth_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
    )

    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)

    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

    for (x, y, w, h) in mouth:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)    
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



img_rgb = facialFeatures('noahfaces/2000.jpg')

plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')

plt.show()