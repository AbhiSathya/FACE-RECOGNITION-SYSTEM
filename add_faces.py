import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('C:/Project/haarcascade_frontalface_default.xml')  # Use forward slashes in the file path

faces_data = []
i = 0

name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    if not ret:  # Check if video capture was successful
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break
video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(-1, 50 * 50 * 3)  # Reshape to a flat array

data_dir = 'C:/Project/data/'

if 'names.pkl' not in os.listdir(data_dir):
    names = [name] * 100
    with open(os.path.join(data_dir, 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)
else:
    with open(os.path.join(data_dir, 'names.pkl'), 'rb') as f:
        names = pickle.load(f)
    names += [name] * 100
    with open(os.path.join(data_dir, 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir(data_dir):
    with open(os.path.join(data_dir, 'faces_data.pkl'), 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(os.path.join(data_dir, 'faces_data.pkl'), 'rb') as f:
        existing_faces = pickle.load(f)
    combined_faces = np.concatenate((existing_faces, faces_data), axis=0)
    with open(os.path.join(data_dir, 'faces_data.pkl'), 'wb') as f:
        pickle.dump(combined_faces, f)
