import cv2
import numpy as np
import pickle
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import time

with open("model.p", "rb") as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

root = tk.Tk()
root.title("Sign Language Recognition")

video_label = tk.Label(root)
video_label.pack()

letter_label = tk.Label(root, text="Detected Letter: ", font=("Arial", 20))
letter_label.pack()

sentence_display = tk.Text(root, height=2, width=50, font=("Arial", 16))
sentence_display.pack()

def reset_sentence():
    sentence_display.delete("1.0", tk.END)

reset_button = tk.Button(root, text="Reset", command=reset_sentence, font=("Arial", 16))
reset_button.pack()

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

last_detected_time = None
last_registered_time = 0
current_letter = ""
sentence = ""

def update_frame():
    global last_detected_time, last_registered_time, current_letter, sentence
    success, img = cap.read()
    if not success:
        root.after(10, update_frame)
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    data_aux = []
    x_, y_ = [], []
    detected_letter = ""
    rect_filled = False
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)
        
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))
        
        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_index = int(prediction[0])
            detected_letter = chr(predicted_index + 65)
            
            if last_detected_time is None:
                last_detected_time = time.time()
            elif time.time() - last_detected_time >= 1.5:
                current_letter = detected_letter
                sentence += current_letter
                sentence_display.insert(tk.END, current_letter)
                last_detected_time = None
                last_registered_time = time.time()
                rect_filled = True
        
        x1, y1 = int(min(x_) * img.shape[1]) - 10, int(min(y_) * img.shape[0]) - 10
        x2, y2 = int(max(x_) * img.shape[1]) + 10, int(max(y_) * img.shape[0]) + 10
        
        if rect_filled or (time.time() - last_registered_time <= 0.1):
            color = (0, 255, 255)
            thickness = -1
        else:
            color = (0, 255, 255)
            thickness = 2
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    else:
        last_detected_time = None
    
    letter_label.config(text=f"Detected Letter: {detected_letter}")
    
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img
    video_label.configure(image=img)
    
    root.after(10, update_frame)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
