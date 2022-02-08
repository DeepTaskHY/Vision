#!/usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import face_recognition
import json
import numpy as np
import os
import rospy
import time
from cv_bridge import CvBridge, CvBridgeError
from dtroslib.helpers import get_package_path
from sensor_msgs.msg import CompressedImage, CameraInfo
from std_msgs.msg import String
import threading
from glob import glob

test_path = get_package_path('vision')
# test_path = '..'

_count = 0


class FaceRecognizer:
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.camera = VideoCamera()

        self.known_face_encodings = []
        self.known_face_ids = []

        # Load sample pictures and learn how to recognize it.
        self.dirname = test_path + '/data/known_face'
        files = os.listdir(self.dirname)
        for filename in files:
            fid, ext = os.path.splitext(filename)

            if ext != '.jpg':
                continue

            fid = int(fid.split('_')[0])
            pathname = os.path.join(self.dirname, filename)
            img = face_recognition.load_image_file(pathname)
            face_encodings = face_recognition.face_encodings(img)

            if face_encodings:
                self.known_face_ids.append(fid)
                self.known_face_encodings.append(face_encodings[0])

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_ids = []
        self.process_this_frame = True
        self.temp_var = None

    def __del__(self):
        del self.camera

    def register_new_face(self, new_face_img, new_face_encoding):
        new_face_id = max(self.known_face_ids) + 1
        cv2.imwrite(os.path.join(
            self.dirname, f'{new_face_id}.jpg'), new_face_img)
        self.known_face_encodings.append(new_face_encoding)
        self.known_face_ids.append(new_face_id)
        return new_face_id

    def register_face(self, face_img, face_id, face_encoding):
        face_img_files = glob(os.path.join(self.dirname, f'{face_id}_*.jpg'))
        file_cnt = len(face_img_files)
        if file_cnt < 2:
            filename = os.path.join(self.dirname, f'{face_id}_{file_cnt}.jpg')
            cv2.imwrite(filename, face_img)
            self.known_face_ids.append(face_id)
            self.known_face_encodings.append(face_encoding)

    def get_frame(self):
        # Grab a single frame of video
        frame = self.camera.get_frame()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(
                rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(
                rgb_small_frame, self.face_locations)

            self.face_ids = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding)
                min_value = min(distances)
                self.temp_var = min_value
                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                # name = "Unknown"
                if min_value < 0.25:
                    # register known face
                    index = np.argmin(distances)
                    fid = self.known_face_ids[index]
                    self.register_face(frame, fid, face_encoding)
                    # if min_value > 0.1:
                    #     self.register_face(frame, fid, face_encoding)

                    self.face_ids.append(fid)
                elif min_value > 0.6:
                    fid = len(set(self.known_face_ids))+1
                    self.register_face(frame, fid, face_encoding)
                    self.face_ids.append(fid)
                else:
                    pass

        self.process_this_frame = not self.process_this_frame

        # Display the results
        for (top, right, bottom, left), fid in zip(self.face_locations, self.face_ids):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35),
                          (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, 'Face ID [{}], {}'.format(
                fid, self.temp_var), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame, self.face_ids

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(int(os.environ['CAMERA_INDEX']))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        # Grab a single frame of video
        ret, frame = self.video.read()
        return frame


def to_ros_msg(data):
    global _count

    json_msg = {
        'header': {
            'source': 'vision',
            'target': ['planning'],
            'content': 'face_recognition',
            'id': _count+1,
            'timestamp': str(time.time())
        },
        'face_recognition': {
            'face_id': data,
            'timestamp': str(time.time())
        }
    }
    ros_msg = json.dumps(json_msg, ensure_ascii=False, indent=4)

    return ros_msg


if __name__ == '__main__':
    global _get_fid
    rospy.init_node('vision_node')
    rospy.loginfo('Start Vision')
    publisher = rospy.Publisher('/recognition/face_id', String, queue_size=10)
    img_pub = rospy.Publisher(
        "/recognition/image/compressed", CompressedImage, queue_size=10)
    bridge = CvBridge()

    fr = FaceRecognizer()

    rate = rospy.Rate(fr.camera.fps)

    while not rospy.is_shutdown():
        frame, face_names = fr.get_frame()

        try:
            img_msg = bridge.cv2_to_compressed_imgmsg(frame, "jpg")
            img_pub.publish(img_msg)
        except CvBridgeError as err:
            print(err)

        try:
            publisher.publish(to_ros_msg(face_names[0]))
        except:
            pass

        rate.sleep()
