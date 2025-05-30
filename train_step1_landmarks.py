import os, pickle, argparse
from utils.proc_vid import parse_vid, resize_vid
from utils.face_proc import FaceProc
import cv2

def main(args):

    VISUALIZE_DIRECT = False # Set to 'False' if you don't want to visualize directly landmarks
    VISUALIZE_DOWNLOAD = False # Set to 'False' if you don't want to download one frame with landmarks visualization every 10 frames

    video_dir_dict = {
        'real': args.real_video_dir,
        'fake': args.fake_video_dir
    }

    face_inst = FaceProc()

    # Load previous results if they exist
    if os.path.exists(args.output_landmark_path):
        with open(args.output_landmark_path, 'rb') as f:
            info_dict = pickle.load(f)
    else:
        info_dict = {}

    for tag in video_dir_dict:

        vid_list = [v for v in os.listdir(video_dir_dict[tag]) if v.endswith('.mp4')]
        print(vid_list)
        for vid_name in vid_list:
            vid_path = os.path.join(video_dir_dict[tag], vid_name)

            new_vid_path = resize_vid(vid_path)

            # Skip if already processed
            if new_vid_path in info_dict:
                print(f"Already processed: {new_vid_path}, skipping.")
                continue

            print(f'Processing video: {new_vid_path}')
            try:
                info = {}
                frames, frame_num, fps, width, height = parse_vid(new_vid_path)
                info['label'] = tag
                info['height'] = height
                info['width'] = width
                info['fps'] = fps
                #info['frames'] = frames  # We need that to visualize the headposes during Step 2
                info['frame_num'] = frame_num

                mark_list_all = []
                for i, img in enumerate(frames):
                    if i<100:
                        try:
                            landmarks = face_inst.get_landmarks(img)
                            if landmarks is None:
                                print(f"Frame {i}: No face detected in {new_vid_path}")
                                continue
                            else:
                                if (VISUALIZE_DIRECT or VISUALIZE_DOWNLOAD) and i % 10 == 0:
                                    central_indices = list(range(17, 36)) + [48, 54]  # 0-indexed: 18–36, 49, 55
                                    full_indices = list(range(0,36)) + [48, 54]
                            
                                    for idx, (x, y) in enumerate(landmarks):
                                        color = (0, 0, 255) if idx in central_indices else (0, 255, 0) if idx in full_indices else (255, 255, 255)
                                        cv2.circle(img, (int(x), int(y)), 2, color, -1)
                                        if VISUALIZE_DIRECT:
                                            cv2.imshow("Landmarks", img)
                                        if VISUALIZE_DOWNLOAD:
                                            cv2.imwrite(f"landmark_viz/3separation/{tag}_{vid_name}_frame{i}.jpg", img)
                                        cv2.waitKey(1)
                            mark_list_all.append(landmarks)
                        except Exception as e:
                            print(f"Error in frame {i} of {new_vid_path}: {e}")
                            continue
                        landmarks = face_inst.get_landmarks(img)
                        
                info['landmarks'] = mark_list_all
                info_dict[new_vid_path] = info

                # Save after every video
                with open(args.output_landmark_path, 'wb') as f:
                    pickle.dump(info_dict, f)

                print(f"Saved landmarks for {new_vid_path} — {len(mark_list_all)} frames processed.")

                # Clear memory
                del frames, mark_list_all, info

            except Exception as e:
                print(f"Failed to process {new_vid_path}: {e}")
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="headpose forensics: train step 1")
    parser.add_argument('--real_video_dir', type=str, default='data/real')
    parser.add_argument('--fake_video_dir', type=str, default='data/fake')
    parser.add_argument('--output_landmark_path', type=str, default='cache/landmark_info.p')
    args = parser.parse_args()
    main(args)