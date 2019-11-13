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
    fps = video_capture.get(cv2.CAP_PROP_FPS)
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
        frame_emotion_dict[frame_number/float(fps)] = current_emotion
        frame_number+=1
    save_file_name = os.path.basename(video_filename)
    save_file_name += 'face_emotion.json'
    with open(save_file_name, 'w') as fp:
        json.dump(frame_emotion_dict, fp)



if __name__ == '__main__':
    video_filename = '/home/sunny/semester 7/multimedia/Project/Dataset/sunflower.mp4'
    face_emotion_analysis(video_filename)