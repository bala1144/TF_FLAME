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

    dataset_path = os.path.join(os.getenv('HOME'), "projects/dataset/voca")
    seq_name = dataset_file[:dataset_file.find(".pkl")]
    outfile_path = os.path.join(dataset_path, "voca_flame_fitting", seq_name)
    os.makedirs(outfile_path, exist_ok=True)

    dataset_file = os.path.join(dataset_path, dataset_file)
    loaded_data = pickle.load(open(dataset_file, 'rb'), encoding="latin1")

    # mesh_render = Facerender()
    flame_config["batch_size"] = 1
    flame_config["flame_model_path"] = os.path.join(os.getenv('HOME'), "projects/TF_FLAME/FLAMEModel", "model/generic_model.pkl")
    flamelayer = FLAME(flame_config)
    flames_faces = flamelayer.faces
    if torch.cuda.is_available():
        flamelayer.cuda()

    fitted_results = {}
    for seq, seq_dict in loaded_data.items():

        if 'len' in seq:
            continue
        
        print("\nRunning fitting on seqs", seq)
        gt_mesh = seq_dict["mesh"]
        mesh_predictions = []
        result_meshes = []

        # create seq_out folder
        # this is used for rendering
        seq_out = os.path.join(outfile_path, seq)
        os.makedirs(seq_out, exist_ok=True)

        previous_state_variable = None

        num_verts = gt_mesh.shape[0]
        import random
        sample_idx = list(random.sample(range(0, num_verts), 5))
        sample_idx.extend([0,10,20])

        for f_id in range(gt_mesh.shape[0]):
            print("\nrunning on frame", f_id)
            
            with Stopwatch("Fitting time") as stopwatch:
                result_vertices, pose, rot, trans, shape, exprs, previous_state_variable = \
                    fit_3D_mesh_with_init(gt_mesh[f_id], init_state=previous_state_variable)
            
            mesh_predictions.append([pose, rot, trans, shape, exprs])
            result_meshes.append(result_vertices)

            # if f_id in sample_idx:
            #     face_with_exprs = get_flame_face_given_expression(flamelayer, exprs)[0]
            #     # render the images
            #     write_image(mesh_render, result_vertices, gt_mesh[f_id], flames_faces, seq_out, f_id, face_with_exprs)

        # store them as video and images
        fitted_results[seq] = mesh_predictions

        # store results for this particular seq
        current_seq_results = {seq : mesh_predictions}
        current_seq_out_file = os.path.join(outfile_path, seq+".pkl")
        pickle.dump(current_seq_results, open(current_seq_out_file, "wb"))


    # store the fitted results
    out_file = os.path.join(outfile_path, "fitted_params.pkl")
    pickle.dump(fitted_results, open(out_file, "wb"))


def write_image(mesh_render, result_vertices, gt_vertices, flames_faces, seq_out, f_id, face_with_exprs=None):

    mesh_render.add_face(result_vertices, flames_faces)
    fitted_mesh_image = mesh_render.render()
    cv2.putText(fitted_mesh_image, 'fitted', (20, 60), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    mesh_render.add_face(gt_vertices, flames_faces)
    gt_image = mesh_render.render()
    cv2.putText(gt_image, 'GT', (20, 60), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # render the mesh
    final_image = np.concatenate([gt_image, fitted_mesh_image], axis=1)

    if face_with_exprs is not None:
        mesh_render.add_face(face_with_exprs, flames_faces)
        face_with_exprs_image = mesh_render.render()
        cv2.putText(face_with_exprs_image, 'Flame_exprs_only', (20, 60), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # render the mesh
        final_image = np.concatenate([final_image, face_with_exprs_image], axis=1)

    file_out = os.path.join(seq_out, "frame%05d.png" % f_id)
    print("Storing rendered files", file_out)
    cv2.imwrite(file_out, final_image)

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

