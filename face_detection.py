import cv2
import threading
import winsound
import time

detected_person = False

def welcome():
    print("Heh")
    while True:
        if detected_person == True:
            print("Heh")
            winsound.PlaySound('face_recognition\welcome_soundeffect.wav', winsound.SND_FILENAME)
            time.sleep(10)

threading.Thread(target=welcome,daemon=True).start()

video = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier('face_recognition\haarcascade_frontalface_default.xml')

while True:
    ret,frame = video.read()
    if not ret:
        print("Image not captured properly")
        break
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_detect.detectMultiScale(gray_frame,1.3,5)

    print(detected_person)
    if type(faces) is tuple:
        detected_person = False
    else:
        detected_person = True
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

    cv2.putText(frame, "Face Recognition", (180,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.namedWindow("Face_Detection",cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Face_Detection",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Face_Detection",frame)
    if ord('q') == cv2.waitKey(1):
        break
video.release()
cv2.destroyAllWindows()