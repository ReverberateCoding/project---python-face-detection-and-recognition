import threading
import winsound

def welcome():
    while True:
        print("Test")
        winsound.PlaySound('face_recognition\welcome_soundeffect.wav', winsound.SND_FILENAME)

threading.Thread(target=welcome,daemon=True).start()

input("nt")
