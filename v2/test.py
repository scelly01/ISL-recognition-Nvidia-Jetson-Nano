import mediapipe as mp
import cv2
import time
import numpy as np
import os


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])




actions = np.array(['accident', 'call', 'doctor', 'help', 'hot', 'lose', 'pain', 'thief'])

no_sequences = 30
sequence_length = 30
start_folder = 0


root_path = r"C:\Users\Sparsh Celly\Desktop\isl_sem2_1.1\MP_Data2"





### COLLECTING DATA ###
#
#  for i in range(0,30):
#      path = os.path.join(root_path, str(i))
#      os.mkdir(path)
#
#
#
# cap = cv2.VideoCapture(0)
# # Set mediapipe model
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     # NEW LOOP
#     # Loop through actions
#     #for action in actions:
#         # Loop through sequences aka videos
#     action = "thief"
#     for sequence in range(start_folder, start_folder + no_sequences):
#         # Loop through video length aka sequence length
#         for frame_num in range(sequence_length):
#
#             # Read feed
#             ret, frame = cap.read()
#
#             # Make detections
#             image, results = mediapipe_detection(frame, holistic)
#
#             # Draw landmarks
#             draw_styled_landmarks(image, results)
#
#             # NEW Apply wait logic
#             if frame_num == 0:
#                 cv2.putText(image, 'STARTING COLLECTION', (120, 200),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
#                 cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                 # Show to screen
#                 cv2.imshow('OpenCV Feed', image)
#                 cv2.waitKey(3000)
#             else:
#                 cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#
#                 cv2.imshow('OpenCV Feed', image)
#
#             # NEW Export keypoints
#             keypoints = extract_keypoints(results)
#             npy_path = os.path.join(root_path, action, str(sequence), str(frame_num))
#             np.save(npy_path, keypoints)
#
#             # Break gracefully
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break
#
#     cap.release()
#     cv2.destroyAllWindows()
#



### CREATING MODEL ###
#
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
#
#
# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))
#
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#
#
#
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
#
# label_map = {label:num for num, label in enumerate(actions)}
#
# sequences, labels = [], []
# for action in actions:
#     for sequence in np.array(os.listdir(os.path.join(root_path, action))).astype(int):
#         window = []
#         for frame_num in range(sequence_length):
#             res = np.load(os.path.join(root_path, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])
#
# print(np.array(sequences).shape)
#


### COMPILING MODEL ###
#
# X = np.array(sequences)
# Y = to_categorical(labels).astype(int)
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model.fit(X, Y, epochs=20)
# #
# print(model.summary())
# #
# model.save('finalmodel2.h5')





### TESTING REAL TIME ###

import tensorflow
model = tensorflow.keras.models.load_model(r"C:\Users\Sparsh Celly\Desktop\isl_sem2_1.1\finalmodel1.h5")


sequence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)
        #print(results)

        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

        cv2.imshow('OpenCV Feed', image)

        # Break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


