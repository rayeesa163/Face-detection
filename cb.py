import cv2

# Load the Haar cascade for face detection
haar_cascade = cv2.CascadeClassifier(r"C:\Users\sraye\Downloads\haarcascade-frontalface-default.xml")
if haar_cascade.empty():
    print("Error loading cascade file")
    exit()

# Initialize the camera (0 for default camera, 1 for external camera)
cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    cv2.imshow("Face Detection", img)

    key = cv2.waitKey(10)
    if key == 27:  # ESC key to exit
        break

cam.release()
cv2.destroyAllWindows()
