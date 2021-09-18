from flask import Flask, render_template, jsonify, session
from flask_socketio import SocketIO, emit
import base64
import os
import time
import uuid
import cv2
import io
import PIL.Image 
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

app = Flask(__name__)
app.config['SECRET_KEY'] = 'henrynamnguyen'
socketio = SocketIO(app,max_http_buffer_size=5000000)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.save_weights('action2.h5')

actions = np.array(['hello', 'thanks', 'I love you'])

mp_holistic = mp.solutions.holistic #Holistic model
mp_drawing = mp.solutions.drawing_utils #Drawing model
holistic = mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5)

threshold = 0.7

# ML helper functions
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

# Util helper functions
def save_to_session(key_name, value):
    if key_name not in session:
        session[key_name] = []
        temp = session[key_name]
        temp.append(value)
    else:
        temp = session[key_name]
        temp.append(value)

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@socketio.on('connect')
def connect():
    print('Client connected')

@socketio.on('image')
def image(data):
    print('Image received before decoding')
    data_without_header = data.split(',',1)[1]
    img_string = base64.b64decode(data_without_header)

    frame = cv2.imdecode(np.fromstring(img_string, dtype=np.uint8), cv2.IMREAD_COLOR)

    frame, results = mediapipe_detection(frame, holistic)
    print('results are ', results)

    # keypoints prediction logic
    keypoints = extract_keypoints(results)
    print('keypoints are ',keypoints)
    save_to_session('sequence', keypoints)
    sequence = session['sequence']
    print('Before Length of sequence is ', len(sequence))
    sequence = sequence[-30:]
    print('After Length of sequence is ', len(sequence))

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        print(actions[np.argmax(res)])
        emit('response_back',actions[np.argmax(res)])
    """
    # viz logic
    # sentence = session['sentence']
        predictions = session['predictions']
        if 'sentence' not in session:
            session['sentence'] = []
            sentence = session['sentence']
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold:
                    save_to_session('sentence', actions[np.argmax(res)])
            
            emit('response_back', sentence)

        sentence = session['sentence']
        if np.unique(predictions[-10:])[0]==np.argmax(res): 
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        # sentence.append(actions[np.argmax(res)])
                        save_to_session('sentence', actions[np.argmax(res)])
                else:
                    # sentence.append(actions[np.argmax(res)])
                    save_to_session('sentence', actions[np.argmax(res)])

        if len(sentence) > 5: 
            sentence = sentence[-5:]

        emit('response_back', sentence)
    """
if __name__ =='__main__':
    socketio.run(app, debug=True)