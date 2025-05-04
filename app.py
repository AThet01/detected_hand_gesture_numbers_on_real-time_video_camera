import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

st.title("🖐️ Real-Time Finger Counting")
st.write ("This is detecting gesture Real-Time fingers counting app.")
run = st.button('▶️ Start Camera')

FRAME_WINDOW = st.image([])

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
finger_tips = [4, 8, 12, 16, 20]

def count_fingers(hand_landmarks, hand_label):
    count = 0
    # Thumb
    if hand_label == "Right":
        if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_tips[0] - 1].x:
            count += 1
    else:  # Left hand
        if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_tips[0] - 1].x:
            count += 1
    # Other fingers
    for tip_id in finger_tips[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            count += 1
    return count

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access camera.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    total_fingers = 0

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            fingers = count_fingers(hand_landmarks, hand_label)
            total_fingers += fingers
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.rectangle(frame, (0, 0), (200, 100), (0, 0, 0), -1)
    cv2.putText(frame, f'Count: {total_fingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
