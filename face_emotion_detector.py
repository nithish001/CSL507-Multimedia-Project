#taken from https://www.analyticsvidhya.com/blog/2018/12/introduction-face-detection-video-deep-learning-python/

import cv2
import face_recognition
from fer import FER
import json
import os


def get_emotion_face(detector,frame):
    
    face_emotions = detector.detect_emotions(frame)
    if(len(face_emotions) == 0):
        return None
    emotion_values = []
    for face in face_emotions:
        emotions = face['emotions']
        emotion_values.append(max(emotions.keys(), key=(lambda k: emotions[k])))
    
    return max(set(emotion_values), key = emotion_values.count)

def detect_face(frame):
    face_locations = face_recognition.face_locations(frame)
    if(len(face_locations) == 0):
        return 0
    return 1

def face_emotion_analysis(video_filename):
    video_capture = cv2.VideoCapture(video_filename)
    emotion_detector = FER()
    frame_number = 1
    frame_emotion_dict = {}
    while True:
        print(frame_number)
        ret, frame = video_capture.read()
        if(ret == False):
            break
        rgb_frame = frame[:,:,::-1]
        face_detect = detect_face(rgb_frame)
        current_emotion = None
        if(face_detect == 1):
            current_emotion = get_emotion_face(emotion_detector, rgb_frame)
        frame_emotion_dict[frame_number] = current_emotion
        frame_number+=1
    save_file_name = os.path.basename(video_filename)
    save_file_name += 'face_emotion.json'
    with open(save_file_name, 'w') as fp:
        json.dump(frame_emotion_dict, fp)




# haar_file = 'haarcascade_frontalface_default.xml'
# # Get a reference to webcam 
# video_capture = cv2.VideoCapture("/home/sunny/semester 7/multimedia/Project/Dataset/sunflower.mp4")
# face_cascade = cv2.CascadeClassifier(haar_file) 
# # Initialize variables
# face_locations = []
# detector = FER()
# while True:
#     # Grab a single frame of video
#     ret, frame = video_capture.read()

#     # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#     rgb_frame = frame[:, :, ::-1]

#     # Find all the faces in the current frame of video
    
#     face_locations = face_cascade.detectMultiScale(frame, 1.3, 4)
#     # for (x, y, w, h) in faces: 
#     #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
#     #face_locations = face_recognition.face_locations(rgb_frame)
#     # #print(face_locations)
#     # # # Display the results
#     # # for top, right, bottom, left in face_locations:
#     #     # Draw a box around the face
        
#     if(len(face_locations) > 0):
#         for (x, y, w, h) in face_locations:
#             emotion = get_emotion_face(detector, frame)
#             print('emotion is ', emotion)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#     # Display the resulting image
            
#     cv2.imshow('Video', frame)
#     # Hit 'q' on the keyboard to quit!
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()

if __name__ == '__main__':
    video_filename = '/home/sunny/semester 7/multimedia/Project/Dataset/sunflower.mp4'
    face_emotion_analysis(video_filename)