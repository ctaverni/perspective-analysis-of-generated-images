import os
from utils.face_proc import FaceProc
import argparse
import pickle
from forensic_test import exam_img, exam_video
import numpy as np


def main(args):
    all_paths = os.listdir(args.input_dir)
    proba_list = []

    # initiate face process class, used to detect face and extract landmarks
    face_inst = FaceProc()

    # initialize SVM classifier for face forensics
    with open(args.classifier_path, 'rb') as f:
        model = pickle.load(f)
    classifier = model[0]
    scaler = model[1]

    all_probas = []
    nb_optout = 0
    for f_name in all_paths:
        f_path = os.path.join(args.input_dir, f_name)
        print('_'*20)
        print('Testing: ' + f_name)
        suffix = f_path.split('.')[-1]
        if suffix.lower() in ['jpg', 'png', 'jpeg', 'bmp']:
            proba, optout = exam_img(args, f_path, face_inst, classifier, scaler)
        elif suffix.lower() in ['mp4', 'avi', 'mov', 'mts']:
            proba, optout = exam_video(args, f_path, face_inst, classifier, scaler)
        print('fake_proba: {},   optout: {}'.format(str(proba), optout))
        tmp_dict = dict()
        tmp_dict['file_name'] = f_name
        tmp_dict['probability'] = proba
        tmp_dict['optout'] = optout
        proba_list.append(tmp_dict)
        if not optout:
            all_probas.append(proba)
        else:
            nb_optout += 1
    pickle.dump(proba_list, open(args.save_file, 'wb'))

    print()
    print('_'*20)
    print()
    print("The mean of all probas is ", np.mean(all_probas))
    nb_fakes_5 = sum(1 for x in all_probas if x > 0.5)
    print(f"(thr=0.5) The nb of 'fake' detected videos is {nb_fakes_5} out of {len(all_probas)} ({nb_fakes_5/len(all_probas)*100}%)")
    nb_fakes_6 = sum(1 for x in all_probas if x > 0.6)
    print(f"(thr=0.6) The nb of 'fake' detected videos is {nb_fakes_6} out of {len(all_probas)} ({nb_fakes_6/len(all_probas)*100}%)")
    nb_fakes_7 = sum(1 for x in all_probas if x > 0.7)
    print(f"(thr=0.7) The nb of 'fake' detected videos is {nb_fakes_7} out of {len(all_probas)} ({nb_fakes_7/len(all_probas)*100}%)")
    nb_fakes_8 = sum(1 for x in all_probas if x > 0.8)
    print(f"(thr=0.8) The nb of 'fake' detected videos is {nb_fakes_8} out of {len(all_probas)} ({nb_fakes_8/len(all_probas)*100}%)")
    nb_fakes_9 = sum(1 for x in all_probas if x > 0.9)
    print(f"(thr=0.9) The nb of 'fake' detected videos is {nb_fakes_9} out of {len(all_probas)} ({nb_fakes_9/len(all_probas)*100}%)")
    print(f"The number of optout is {nb_optout} ({nb_optout/len(all_paths)*100}%)")


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="headpose forensics")
   parser.add_argument('--input_dir', type=str, default='debug_data')
   parser.add_argument('--markID_c', type=str, default='18-36,49,55', help='landmark ids to estimate CENTRAL face region')
   parser.add_argument('--markID_a', type=str, default='1-36,49,55', help='landmark ids to estimate WHOLE face region')
   parser.add_argument('--classifier_path', type=str, default='models/trained_models/R_full_mat_t_vec_model.p')
   parser.add_argument('--save_file', type=str, default='proba_list.p')
   args = parser.parse_args()
   main(args)