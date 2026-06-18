import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

from sklearn.svm import SVC

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

X = []
y = []

dataset_path = "dataset"

print("Reading dataset...")

for label in os.listdir(dataset_path):

    folder = os.path.join(dataset_path, label)

    if not os.path.isdir(folder):
        continue

    print(f"Processing {label}...")

    for image_name in os.listdir(folder):

        image_path = os.path.join(folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]

            data = []

            for lm in hand.landmark:
                data.append(lm.x)
                data.append(lm.y)

            X.append(data)
            y.append(label)

print("Training SVM...")

model = SVC(kernel="linear")

model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully as model.pkl")