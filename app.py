# import streamlit as st
# from keras.preprocessing.image import img_to_array
# from keras.models import load_model
# import cv2
# import numpy as np
# from keras.preprocessing import image
# import pandas as pd
# from IPython.display import display

# # Suppress warnings
# import warnings
# warnings.filterwarnings("ignore")

# # Load emotion detection model
# model = load_model("ferNet.h5")

# # Load face cascade classifier
# face_haar_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load music dataset
# mood_music = pd.read_csv(
#     "data_moods.csv")
# mood_music = mood_music[['name', 'artist', 'mood']]

# # Function to detect emotion and suggest music


# def detect_emotion(test_img):
#     # Convert image to grayscale
#     gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the grayscale image
#     faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

#     for (x, y, w, h) in faces_detected:
#         # Draw rectangle around the face
#         cv2.rectangle(test_img, (x, y), (x + w, y + h),
#                       (255, 0, 0), thickness=7)

#         # Extract face region
#         roi_gray = gray_img[y:y + w, x:x + h]
#         roi_gray = cv2.resize(roi_gray, (48, 48))
#         img_pixels = image.img_to_array(roi_gray)
#         img_pixels = np.expand_dims(img_pixels, axis=0)
#         img_pixels /= 255

#         # Predict emotion
#         predictions = model.predict(img_pixels)
#         max_index = np.argmax(predictions[0])
#         emotions = ('angry', 'disgust', 'fear', 'happy',
#                     'sad', 'surprise', 'neutral')
#         detected_emotion = emotions[max_index]

#         # Display the detected emotion on the Streamlit UI
#         st.subheader(f"Detected Emotion: {detected_emotion}")

#         # Suggest music based on detected emotion
#         suggest_music(detected_emotion)

#     return test_img


# def suggest_music(emotion):
#     if emotion in ['angry', 'disgust', 'fear']:
#         filter_condition = mood_music['mood'] == 'Calm'
#     elif emotion in ['happy', 'neutral']:
#         filter_condition = mood_music['mood'] == 'Happy'
#     elif emotion == 'sad':
#         filter_condition = mood_music['mood'] == 'Sad'
#     elif emotion == 'surprise':
#         filter_condition = mood_music['mood'] == 'Energetic'
#     else:
#         # Handle other cases as needed
#         return

#     filtered_music = mood_music[filter_condition]
#     if not filtered_music.empty:
#         suggested_music = filtered_music.sample(n=5)
#         suggested_music.reset_index(inplace=True)
#         st.subheader("Suggested Music:")
#         st.table(suggested_music)
#     else:
#         st.warning(f"No music found for emotion: {emotion}")


# def main():
#     st.title("Facial Emotion Analysis and Music Suggestion")

#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Error capturing video feed.")
#             break

#         # Display the video feed with detected emotion
#         st.image(detect_emotion(frame), channels="BGR", use_column_width=True)

#         # Stop the video feed when the user closes the Streamlit app
#         # if st.button("Stop"):
#         #     break

#     # Release the video capture
#     cap.release()


# if __name__ == "__main__":
#     main()


import streamlit as st
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing import image
import pandas as pd
from IPython.display import display

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load emotion detection model
model = load_model("ferNet.h5")

# Load face cascade classifier
face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load music dataset
mood_music = pd.read_csv(
    "data_moods.csv")
mood_music = mood_music[['name', 'artist', 'mood']]

# Function to detect emotion and suggest music


def detect_emotion(test_img):
    # Convert image to grayscale
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        # Draw rectangle around the face
        cv2.rectangle(test_img, (x, y), (x + w, y + h),
                      (255, 0, 0), thickness=7)

        # Extract face region
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        # Predict emotion
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy',
                    'sad', 'surprise', 'neutral')
        detected_emotion = emotions[max_index]

        # Display the detected emotion on the Streamlit UI
        st.subheader(f"Detected Emotion: {detected_emotion}")

        # Suggest music based on detected emotion
        suggest_music(detected_emotion)

    return test_img


def suggest_music(emotion):
    if emotion in ['angry', 'disgust', 'fear']:
        filter_condition = mood_music['mood'] == 'Calm'
    elif emotion in ['happy', 'neutral']:
        filter_condition = mood_music['mood'] == 'Happy'
    elif emotion == 'sad':
        filter_condition = mood_music['mood'] == 'Sad'
    elif emotion == 'surprise':
        filter_condition = mood_music['mood'] == 'Energetic'
    else:
        # Handle other cases as needed
        return

    filtered_music = mood_music[filter_condition]
    if not filtered_music.empty:
        suggested_music = filtered_music.sample(n=5)
        suggested_music.reset_index(inplace=True)
        st.subheader("Suggested Music:")
        st.table(suggested_music)
    else:
        st.warning(f"No music found for emotion: {emotion}")


def main():
    st.title("Facial Emotion Analysis and Music Suggestion")

    cap = cv2.VideoCapture(0)
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error capturing video feed.")
            break

        # Display the video feed with detected emotion
        st.image(detect_emotion(frame), channels="BGR", use_column_width=True)

        frame_counter += 1
        if frame_counter >= 5:
            break

    # Release the video capture
    cap.release()


if __name__ == "__main__":
    main()
