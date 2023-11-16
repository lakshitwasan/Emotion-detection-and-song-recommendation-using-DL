# # from keras.preprocessing.image import load_img, img_to_array
# # import matplotlib.pyplot as plt
# # from keras.models import load_model
# # import os
# # import cv2
# # import numpy as np
# # from keras.preprocessing import image
# # import warnings
# # warnings.filterwarnings("ignore")

# # # load model
# # model = load_model("ferNet.h5")


# # face_haar_cascade = cv2.CascadeClassifier(
# #     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# # cap = cv2.VideoCapture(0)

# # while True:
# #     # captures frame and returns boolean value and captured image
# #     ret, test_img = cap.read()
# #     if not ret:
# #         continue
# #     gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# #     faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

# #     for (x, y, w, h) in faces_detected:
# #         cv2.rectangle(test_img, (x, y), (x + w, y + h),
# #                       (255, 0, 0), thickness=7)
# #         # cropping region of interest i.e. face area from  image
# #         roi_gray = gray_img[y:y + w, x:x + h]
# #         roi_gray = cv2.resize(roi_gray, (48, 48))
# #         img_pixels = image.img_to_array(roi_gray)
# #         img_pixels = np.expand_dims(img_pixels, axis=0)
# #         img_pixels /= 255

# #         predictions = model.predict(img_pixels)

# #         # find max indexed array
# #         max_index = np.argmax(predictions[0])

# #         emotions = ('angry', 'disgust', 'fear', 'happy',
# #                     'sad', 'surprise', 'neutral')
# #         predicted_emotion = emotions[max_index]

# #         cv2.putText(test_img, predicted_emotion, (int(x), int(y)),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# #     resized_img = cv2.resize(test_img, (1000, 700))
# #     cv2.imshow('Facial emotion analysis ', resized_img)

# #     if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
# #         break

# # cap.release()
# # cv2.destroyAllWindows


# from keras.preprocessing.image import img_to_array
# import matplotlib.pyplot as plt
# from keras.models import load_model
# import os
# import cv2
# import numpy as np
# from keras.preprocessing import image
# import warnings

# warnings.filterwarnings("ignore")

# # load model
# model = load_model("ferNet.h5")

# face_haar_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# cap = cv2.VideoCapture(0)

# while True:
#     ret, test_img = cap.read()
#     if not ret:
#         continue

#     gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

#     faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

#     for (x, y, w, h) in faces_detected:
#         cv2.rectangle(test_img, (x, y), (x + w, y + h),
#                       (255, 0, 0), thickness=7)

#         roi_gray = gray_img[y:y + w, x:x + h]
#         roi_gray = cv2.resize(roi_gray, (48, 48))
#         img_pixels = image.img_to_array(roi_gray)
#         img_pixels = np.expand_dims(img_pixels, axis=0)
#         img_pixels /= 255

#         predictions = model.predict(img_pixels)

#         max_index = np.argmax(predictions[0])
#         emotions = ('angry', 'disgust', 'fear', 'happy',
#                     'sad', 'surprise', 'neutral')
#         predicted_emotion = emotions[max_index]

#         cv2.putText(test_img, predicted_emotion, (int(x), int(y)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     resized_img = cv2.resize(test_img, (1000, 700))
#     cv2.imshow('Facial emotion analysis ', resized_img)

#     if cv2.waitKey(10) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from keras.models import load_model
import os
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
mood_music = pd.read_csv("data_moods.csv")
mood_music = mood_music[['name', 'artist', 'mood']]

# Open video capture
cap = cv2.VideoCapture(0)


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
        display(suggested_music)
    else:
        print(f"No music found for emotion: {emotion}")


while True:
    ret, test_img = cap.read()
    if not ret:
        continue

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

        # Suggest music based on detected emotion
        suggest_music(detected_emotion)

    # Display the video feed with emotion text
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis', resized_img)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(10) == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
