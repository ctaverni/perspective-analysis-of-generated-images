"""
Face related processing class:
1. Face alignment
2. Face landmarks
3. ...
"""

import dlib
from utils.proc_vid import parse_vid
from utils.face_utils import shape_to_np
import cv2
from ultralytics import YOLO


class FaceProc(object):

    def __init__(self):
        # Initialiser dlib pour les landmarks
        self.landmark_estimatior = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

        # Charger YOLO pour la détection
        self.yolo_model = YOLO("models/yolov11n-face.pt")

    def detect_faces_yolo(self, img):
        # Utilise YOLO pour détecter les visages (classe 'person')
        results = self.yolo_model(img, verbose=False)
        faces = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if self.yolo_model.names[cls_id] == 'face':
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                faces.append((x1, y1, x2, y2))

        return faces

    def get_landmarks(self, img, draw_rect=False):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_rects = self.detect_faces_yolo(img)

        if len(face_rects) == 0:
            return None

        x1, y1, x2, y2 = face_rects[0]
        rect_dlib = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)

        if draw_rect:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        marks = self.landmark_estimatior(img_rgb, rect_dlib)
        marks = shape_to_np(marks)

        return marks

    def get_all_face_rects(self, img):
        return self.detect_faces_yolo(img)

    def get_landmarks_all_faces(self, img, rects):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        all_landmarks = []

        for x1, y1, x2, y2 in rects:
            rect_dlib = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
            marks = self.landmark_estimatior(img_rgb, rect_dlib)
            marks = shape_to_np(marks)
            all_landmarks.append(marks)

        return all_landmarks

    def get_landmarks_vid(self, video_path):
        print('vid_path: ' + video_path)
        imgs, frame_num, fps, width, height = parse_vid(video_path)
        mark_list = []

        for img in imgs:
            mark = self.get_landmarks(img)
            mark_list.append(mark)

        return mark_list

