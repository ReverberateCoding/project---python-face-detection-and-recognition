import cv2
import os

video = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier('face_recognition\haarcascade_frontalface_default.xml')
count = 0
unique = False

while not unique:

    nameID=str(input("Enter Your Name: ")).lower()

    img_path=os.path.join('face_recognition\images',nameID)

    if os.path.exists(img_path):
        print("Name Already Taken")
    else:
        os.makedirs(img_path)
        unique = True

while True:
    ret,frame = video.read()
    if not ret:
        print("Image not captured properly")
        break
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_detect.detectMultiScale(gray_frame,1.3,5)
    for (x, y, w, h) in faces:
        count = count+1
        img_name = os.path.join(img_path,nameID+'_{}.jpg'.format(count))
        print("Creating Image "+'{}'.format(count))
        cv2.imwrite(img_name, frame[y:y+h,x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    cv2.imshow("Data_Collection",frame)
    if ord('q') == cv2.waitKey(1):
        break
    if count >= 500:
        break
video.release()
cv2.destroyAllWindows()


