import argparse, os, pickle
import numpy as np
import cv2
from utils.head_pose_proc import PoseEstimator

def draw_pose_axes(img, camera_matrix, rvec, tvec, length=50):
    axis = np.float32([[length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    origin = np.float32([[0, 0, 0]]).reshape(-1, 3)
    imgpts, _ = cv2.projectPoints(np.vstack((origin, axis)), rvec, tvec, camera_matrix, np.zeros((4,1)))
    p_origin = tuple(imgpts[0].ravel().astype(int))
    p_x = tuple(imgpts[1].ravel().astype(int))
    p_y = tuple(imgpts[2].ravel().astype(int))
    p_z = tuple(imgpts[3].ravel().astype(int))

    cv2.line(img, p_origin, p_x, (0,0,255), 2)
    cv2.line(img, p_origin, p_y, (0,255,0), 2) 
    cv2.line(img, p_origin, p_z, (255,0,0), 2) 

def main(args):
    VISUALIZE_DIRECT = False  # Set to 'False' if you don't want to visualize directly
    VISUALIZE_DOWNLOAD = False  # Set to 'False' if you don't want to download one frame with headposes visualization every 10 frames
    

    with open(args.landmark_info_path, 'rb') as f:
        vids_info = pickle.load(f)

    markID_c = args.markID_c
    markID_a = args.markID_a
    save_pose_file = args.headpose_save_path

    for key, value in vids_info.items():
        print(key)
        landmark_2d = value['landmarks']
        height = value['height']
        width = value['width']
        pose_estimate = PoseEstimator([height, width])

        R_c_list, R_a_list, t_c_list, t_a_list = [], [], [], []
        R_c_matrix_list, R_a_matrix_list = [], []

        for i, landmark_2d_cur in enumerate(landmark_2d):
            R_c, t_c = None, None
            R_a, t_a = None, None
            R_c_matrix, R_a_matrix = None, None

            if landmark_2d_cur is not None:
                R_c, t_c = pose_estimate.solve_single_pose(landmark_2d_cur, markID_c)
                R_a, t_a = pose_estimate.solve_single_pose(landmark_2d_cur, markID_a)

                R_c_matrix = pose_estimate.Rodrigues_convert(R_c)
                R_a_matrix = pose_estimate.Rodrigues_convert(R_a)

                # Visualisation
                if (VISUALIZE_DOWNLOAD or VISUALIZE_DIRECT) and R_c is not None and t_c is not None:
                    if i % 10 == 0:
                        frame = value['frames'][i]
                        draw_pose_axes(frame, pose_estimate.camera_matrix, R_c, t_c)
                        
                        if VISUALIZE_DIRECT:
                            cv2.imshow("Headposes", frame)

                        if VISUALIZE_DOWNLOAD:
                            cv2.imwrite(f"headpose_viz/{value['label']}_{os.path.basename(key)[:-4]}_frame_{i}.jpg", frame)
                        
                        
                        if cv2.waitKey(50) & 0xFF == ord('q'):
                            break

            R_c_list.append(R_c)
            R_a_list.append(R_a)
            t_c_list.append(t_c)
            t_a_list.append(t_a)
            R_c_matrix_list.append(R_c_matrix)
            R_a_matrix_list.append(R_a_matrix)

        value['R_c_vec'] = R_c_list
        value['R_c_mat'] = R_c_matrix_list
        value['t_c'] = t_c_list
        value['R_a_vec'] = R_a_list
        value['R_a_mat'] = R_a_matrix_list
        value['t_a'] = t_a_list

    with open(save_pose_file, 'wb') as f:
        pickle.dump(vids_info, f)

    print('Done!')
    cv2.destroyAllWindows()

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="headpose forensic: train step 2")
   parser.add_argument('--landmark_info_path', type=str, default='cache/landmark_info.p')
   parser.add_argument('--markID_a', type=str, default='1-36,49,55')
   parser.add_argument('--markID_c', type=str, default='18-36,49,55')
   parser.add_argument('--headpose_save_path', type=str, default='cache/headpose_data.p')
   args = parser.parse_args()
   main(args)
