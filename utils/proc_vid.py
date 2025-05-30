import cv2
import numpy as np
import os


def parse_vid(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    imgs = []
    while True:
        success, image = vidcap.read()
        if success:
            imgs.append(image)
        else:
            break

    vidcap.release()
    if len(imgs) != frame_num:
        frame_num = len(imgs)
    return imgs, frame_num, fps, width, height


def extract_video_id(video_folder):
    all_files = sorted(os.listdir(video_folder))
    IDs = []
    for file in all_files:
        IDs.append(file.split('.')[0])
    return IDs

def resize_vid(video_path):
    MAX_SIZE = 500
    NB_FRAMES = 50

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video.")
        exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if (height > MAX_SIZE) or (width > MAX_SIZE):
        scale = min(MAX_SIZE / width, MAX_SIZE / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        start_frame = max(0, (total_frames // 2) - 25)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        new_video_path = video_path[:-4] + "_resized.mp4"
        out = cv2.VideoWriter(new_video_path, fourcc, fps, (new_width, new_height))

        # Lire et écrire 50 frames
        frames_written = 0
        while frames_written < NB_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (new_width, new_height))
            out.write(resized)
            frames_written += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return new_video_path
    
    else:
        return video_path
