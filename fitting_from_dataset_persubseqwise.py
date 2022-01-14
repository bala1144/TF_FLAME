#!/usr/bin/env python3
import numpy as np
import pickle
from fit_3D_mesh_voca import fit_3D_mesh, fit_3D_mesh_with_init
from utils.render_mesh import flame_render, Facerender
from utils.stopwatch import Stopwatch
import cv2
import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
import argparse
from FLAMEModel.FLAME import FLAME
import torch
import copy

font = cv2.FONT_HERSHEY_SIMPLEX

flame_config = {
"shape_params": 0,
"expression_params": 100,
"pose_params": 0,
"use_3D_translation": False,
"optimize_eyeballpose": False,
"optimize_neckpose": False,
"num_worker": 8,
"batch_size": 1  # set this to batch size * num_subwindow_samples * subsample_window_len
}

def get_flame_face_given_expression(flame_model, exprs):

    exprs = torch.from_numpy(exprs).float()
    if torch.cuda.is_available():
        exprs = exprs.cuda()

    vertices = flame_model.morph(exprs).cpu().numpy()
    return vertices

def sequence_specific_fitting(dataset_file):
    print(dataset_file)
    seq_name = dataset_file.rsplit("_",1)[1]
    subj_name = dataset_file.rsplit("_",1)[0]
    print(seq_name, subj_name)

    dataset_path = os.path.join(os.getenv('HOME'), "projects/dataset/voca", "seqwise_spilted_data")
    outfile_path = os.path.join(os.getenv('HOME'), "projects/dataset/voca", "subj_seq_fitting_results", subj_name)
    os.makedirs(outfile_path, exist_ok=True)

    # tracker file
    tracker_file = os.path.join(outfile_path, dataset_file+".txt")

    # extract current subject name
    in_dataset_file = os.path.join(dataset_path, seq_name+".pkl")
    print("Loading data file", in_dataset_file)
    loaded_data = pickle.load(open(in_dataset_file, 'rb'), encoding="latin1")
    seq_dict = loaded_data[dataset_file]

    # extract the mesh
    gt_mesh = seq_dict["mesh"]
    mesh_predictions = []
    previous_state_variable = None

    for f_id in range(gt_mesh.shape[0]):
        print("\nrunning on frame", f_id)
        
        with Stopwatch("Fitting time") as stopwatch:
            result_vertices, pose, rot, trans, shape, exprs, previous_state_variable = \
                fit_3D_mesh_with_init(gt_mesh[f_id], init_state=previous_state_variable)
        
        mesh_predictions.append([result_vertices, pose, rot, trans, shape, exprs])

        with open(tracker_file, "a") as myfile:
            myfile.write("completed %d\n"%f_id)

    # store results for this particular seq
    current_seq_results = {dataset_file : mesh_predictions}
    current_seq_out_file = os.path.join(outfile_path, dataset_file+".pkl")
    print("results are stored to", current_seq_out_file)
    pickle.dump(current_seq_results, open(current_seq_out_file, "wb"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sequence_specific_fitting')
    parser.add_argument(
        "-d",
        "--dataset_file",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="dataset_file",
        )
    args = parser.parse_args()

    sequence_specific_fitting(dataset_file=args.dataset_file)

