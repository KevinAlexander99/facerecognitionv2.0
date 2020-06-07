import face_recognition
import os
import cv2
import pickle
import time
import csv

KNOWN_FACES_DIR = "known_faces"#dataset
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"

video = cv2.VideoCapture(0)#ngambil input dari webcam

print("loading known faces")

known_faces = []
known_names =[]

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        #image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        #encoding = face_recognition.face_encodings(image)[0]

        encoding = pickle.load(open(f"{name}/{filename}", "rb"))
        known_faces.append(encoding)
        known_names.append(name)
if len(known_names) > 0:
    next_id = max(known_names)+1
else:
    next_id = 0


while True:

    ret, image = video.read()#ngubah video jadi image

    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):#ngebandingin sama data yang udh ada
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:#
            match = known_names[results.index(True)]
            print(f"Match found: {match}")
        else:
            match = str(next_id)
            next_id += 1
            known_names.append(match)
            known_faces.append(face_encoding)
            os.mkdir(f"{KNOWN_FACES_DIR}/{match}")
            pickle.dump(face_encoding, open(f"{KNOWN_FACES_DIR}/{match}/{match}-{int(time.time())}.pkl", "wb"))

        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])
        color = [0, 255, 0]
        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)#buat kotak di daerah muka

        #buat label nama
        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2]+22)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, str(match), (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (200,200,200),FONT_THICKNESS)

    cv2.imshow("", image)#nampilin result
    with open('absen.csv', mode='a') as absen_file:
        absen_writee = csv.writer(absen_file)
        if match == "0":
            absen_writee.writerow([str("Kevin")])
        else:
            absen_writee.writerow([str(match)])
    if cv2.waitKey(1) & 0xFF ==ord("q"):#tombol exit dengan q
        break

#release holder di webcam laptop/computer
video.release()
cv2.destroyAllWindows()
