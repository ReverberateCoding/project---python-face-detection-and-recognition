from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2
import numpy as np
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

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("face_recognition\L_keras_model.h5", compile=False)

# Load the labels
class_names = open("face_recognition\L_labels.txt", "r").readlines()

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
        # Resize the raw image into (224-height,224-width) pixels
        face_image = cv2.resize(frame[y:y+h,x:x+w], (224, 224), interpolation=cv2.INTER_AREA)

        # Make the image a numpy array and reshape it to the models input shape.
        face_image = np.asarray(face_image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        face_image = (face_image / 127.5) - 1
        
        # Predicts the model
        prediction = model.predict(face_image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        confidence_score = (np.round(confidence_score * 100))
        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", f'{round(confidence_score,1)}', "%")
        
        name_length = len(class_name)
        print(name_length)
        text1 = class_name[2:name_length-1] + '[' + f'{round(confidence_score,1)}%' + ']'
        print(text1)

        if confidence_score>=95.0:
            cv2.putText(frame, text1, (x-50,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Unknown', (x,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    

    cv2.putText(frame, "Face Recognition", (180,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.namedWindow("Face_Detection",cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Face_Detection",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    
    cv2.imshow("Face_Detection",frame)

    if ord('q') == cv2.waitKey(1):
        break
video.release()
cv2.destroyAllWindows()