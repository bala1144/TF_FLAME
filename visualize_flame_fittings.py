import os
import numpy as np
import pickle
from fit_3D_mesh_voca import fit_3D_mesh
from utils.render_mesh import flame_render, Facerender
from utils.stopwatch import Stopwatch
import glob
import cv2
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from voca_utils.process_voca import Data_binder, Data_generate_subseqs
from voca_utils.voca_data_cfg import get_default_config
import torch
font = cv2.FONT_HERSHEY_SIMPLEX
from FLAMEModel.FLAME import FLAME

# def get_flame_face_given_expression(flame_model:FLAME, exprs, jaw_pose=None,  shape_params=None, neck_pose=None, eye_pose=None ):
#     exprs = torch.from_numpy(exprs).float()
#     if torch.cuda.is_available():
#         exprs = exprs.cuda()

#     if jaw_pose is not None:
#         jaw_pose = torch.from_numpy(jaw_pose).float()
#         if torch.cuda.is_available():
#             jaw_pose = jaw_pose.cuda()
    
#     if shape_params is not None:
#         shape_params = torch.from_numpy(shape_params).float()
#         if torch.cuda.is_available():
#             shape_params = shape_params.cuda()


#     vertices = flame_model.morph(expression_params=exprs,jaw_pose=jaw_pose, shape_params=shape_params).cpu().numpy()
#     return vertices

def get_flame_face_given_expression(flame_model:FLAME, exprs, jaw_pose=None,  shape_params=None, neck_pose=None, eye_pose=None ):
    exprs = torch.from_numpy(exprs).float()
    
    if jaw_pose is not None:
        jaw_pose = torch.from_numpy(jaw_pose).float()
    
    if neck_pose is not None:
        neck_pose = torch.from_numpy(neck_pose).float()
    
    if eye_pose is not None:
        eye_pose = torch.from_numpy(eye_pose).float()


    if shape_params is not None:
        shape_params = torch.from_numpy(shape_params).float()

    vertices = flame_model.morph(expression_params=exprs,
                                jaw_pose=jaw_pose, 
                                neck_pose=neck_pose,
                                eye_pose=eye_pose,
                                shape_params=shape_params).cpu().numpy()
    return vertices
    
def visualize_fittings_on_seq(sub_seq_name, postfix=""):

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
    mesh_render = Facerender()
    flame_config["batch_size"] = 1
    flame_config["flame_model_path"] = os.path.join(os.getenv('HOME'), "projects/TF_FLAME/FLAMEModel", "model/generic_model.pkl")
    flamelayer = FLAME(flame_config)
    flames_faces = flamelayer.faces
    
    dataset_path = os.path.join(os.getenv('HOME'), "projects/dataset/voca")
    # load the data
    out_path = os.path.join(dataset_path, "vis_fittings")
    os.makedirs(out_path, exist_ok=True)

    seq_out_path = os.path.join(out_path, sub_seq_name+postfix)
    os.makedirs(seq_out_path, exist_ok=True)

    # gt frame path
    seq_name = sub_seq_name.rsplit("_",1)[1]
    subj_name = sub_seq_name.rsplit("_",1)[0]
    gt_mesh_path = os.path.join(dataset_path, "seqwise_spilted_data")
    in_dataset_file = os.path.join(gt_mesh_path, seq_name+".pkl")
    loaded_data = pickle.load(open(in_dataset_file, 'rb'), encoding="latin1")
    seq_dict = loaded_data[sub_seq_name]
    gt_mesh = seq_dict["mesh"]

    # extracted frame path
    acc_file = os.path.join(dataset_path, "subj_seq_fitting_results/accumulated_extracted_ns473.pkl")
    loaded_data = pickle.load(open(acc_file, 'rb'), encoding="latin1")
    current_seq = loaded_data[sub_seq_name]
    optim_mesh = current_seq["optim_mesh"]
    exprs = current_seq["exprs"]
    pose = current_seq["pose"]
    shape_params = current_seq["shape"]
    num_verts = optim_mesh.shape[0]
    
    # pose details
    #  neck_pose_reg = tf.reduce_sum(tf.square(tf_pose[:,:3]))
    # jaw_pose_reg = tf.reduce_sum(tf.square(tf_pose[:,3:6]))
    # eyeballs_pose_reg = tf.reduce_sum(tf.square(tf_pose[:,6:]))

    for f_id in range(num_verts):
        print("\nrunning on frame", f_id)
        
        neck_pose = pose[f_id, :3].reshape(1,-1)
        jaw_pose = pose[f_id, 3:6].reshape(1,-1)
        eye_pose = pose[f_id, 6:].reshape(1,-1)

        current_exprs = exprs[f_id].reshape(1,-1)
        current_shape = shape_params[f_id].reshape(1,-1)

        face_with_exprs_shape_pose = get_flame_face_given_expression(flamelayer, current_exprs, jaw_pose, current_shape,
                                                                    neck_pose,
                                                                    eye_pose
                                                                    )[0]
        renderd_face_with_exprs_shape_Fullpose = render_image(mesh_render, face_with_exprs_shape_pose, flames_faces, "face_with_exprs_shape_pose")

        face_with_exprs_shape_jawpose = get_flame_face_given_expression(flamelayer, current_exprs, jaw_pose,current_shape)[0]
        renderd_face_with_exprs_shape_jawpose = render_image(mesh_render, face_with_exprs_shape_jawpose, flames_faces, "face_with_exprs_shape_jawpose")


        face_with_exprs_noshape_jawpose = get_flame_face_given_expression(flamelayer, current_exprs, jaw_pose)[0]
        renderd_face_with_exprs_noshape_jawpose = render_image(mesh_render, face_with_exprs_noshape_jawpose, flames_faces, "face_with_exprs_noshape_jawpose")


        face_with_exprs_noshape_nopose = get_flame_face_given_expression(flamelayer, current_exprs)[0]
        renderd_face_with_exprs_noshape_nopose = render_image(mesh_render, face_with_exprs_noshape_nopose, flames_faces, "face_with_exprs_noshape_nopose")


        # render the images
        renderd_gt_mesh = render_image(mesh_render, gt_mesh[f_id], flames_faces, "gt_mesh")
        renderd_optim_mesh = render_image(mesh_render, optim_mesh[f_id], flames_faces, "fitted_mesh")

        # compose image
        top_row = np.concatenate([renderd_gt_mesh, renderd_optim_mesh, renderd_face_with_exprs_shape_Fullpose], axis=1)
        botton_row = np.concatenate([renderd_face_with_exprs_shape_jawpose, renderd_face_with_exprs_noshape_jawpose, renderd_face_with_exprs_noshape_nopose], axis=1)
        grid = np.concatenate([top_row, botton_row], axis=0)
        file_out = os.path.join(seq_out_path, "frame%05d.png" % f_id)
        print("Storing rendered files", file_out)
        cv2.imwrite(file_out, grid)
    
def render_image(mesh_render, vertices, flames_faces, title):
    mesh_render.add_face(vertices, flames_faces)
    renderd_image = mesh_render.render()
    cv2.putText(renderd_image, title, (20, 60), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return renderd_image


def run_on_accumulate_dataset():
    params = get_default_config()
    seqs_list = params["sequence_for_training"].split()
    subjects = params["all_subjects"].split()

    # for sub in subjects:
    #     for seq in seqs_list:
    #         sub_seq_name = sub+"_"+seq
    #         print("Running on file",  sub_seq_name)
    #         visualize_fittings_on_seq(sub_seq_name)
    #         break
    #     break
    
    for sub in subjects:
        seq = "sentence10"
        sub_seq_name = sub+"_"+seq
        print("Running on file",  sub_seq_name)
        visualize_fittings_on_seq(sub_seq_name)
        print()



if __name__ == "__main__":
    run_on_accumulate_dataset()
    # sub = "FaceTalk_170811_03275_TA_sentence10"
    # visualize_fittings_on_seq(sub, postfix="_grid")