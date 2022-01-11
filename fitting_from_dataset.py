import numpy as np
import pickle
from fit_3D_mesh_voca import fit_3D_mesh
from utils.render_mesh import flame_render, Facerender
from utils.stopwatch import Stopwatch
import glob
import cv2
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from voca_utils.process_voca import Data_binder, Data_generate_subseqs
import trimesh
from voca_utils.voca_data_cfg import get_default_config
import os
import argparse

def sequence_specific_fitting(dataset_file):

    dataset_path = os.path.join(os.getenv('HOME'), "projects/dataset/voca")
    outfile_path = os.path.join(dataset_path, "trainFaceTalk_170904_00128_TA_ns40_fitted_mesh")
    os.makedirs(outfile_path, exist_ok=True)
    font = cv2.FONT_HERSHEY_SIMPLEX


    dataset_file = os.path.join(dataset_path, dataset_file)
    loaded_data = pickle.load(open(dataset_file, 'rb'), encoding="latin1")

    target_mesh_path = os.path.join(os.getenv('HOME'), "projects/TF_FLAME", "data/registered_mesh.ply")
    template_mesh = trimesh.load_mesh(target_mesh_path, process=False)
    flames_faces = template_mesh.faces
    mesh_render = Facerender()

    fitted_results = {}
    for seq, seq_dict in loaded_data.items():

        if 'len' in seq:
            continue
        
        print("Running fitting on seqs", seq)
        gt_mesh = seq_dict["exprs"]
        # run fitting through this and store the get the expressions
        mesh_predictions = []
        result_meshes = []

        # create seq_out folder
        seq_out = os.path.join(outfile_path, seq)
        os.makedirs(seq_out, exist_ok=True)

        # random sample
        num_verts = gt_mesh.shape[0]
        import random
        sample_idx = random.sample(range(0, num_verts), 5)

        # for f_id in range(gt_mesh.shape[0]):
        for f_id in sample_idx:
            print("running on frame", f_id)
            
            with Stopwatch("Fitting time") as stopwatch:
                result_vertices, pose, rot, trans, shape, exprs = fit_3D_mesh(gt_mesh[f_id], flames_faces)
            
            mesh_predictions.append([pose, rot, trans, shape, exprs])
            result_meshes.append(result_vertices)

            mesh_render.add_face(result_vertices, flames_faces)
            fitted_mesh_image = mesh_render.render()
            cv2.putText(fitted_mesh_image, 'fitted', (20, 60), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            mesh_render.add_face(gt_mesh[f_id], flames_faces)
            gt_image = mesh_render.render()
            cv2.putText(gt_image, 'GT', (20, 60), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # render the mesh
            final_image = np.concatenate([gt_image, fitted_mesh_image], axis=1)
            file_out = os.path.join(seq_out, "frame%05d.png"%f_id)
            print("Storing rendered files", file_out)
            cv2.imwrite(file_out, final_image)

        # store them as video and images
        fitted_results[seq] = mesh_predictions

    # store the fitted results
    out_file = os.path.join(outfile_path, "fitted_params.pkl")
    pickle.dump(fitted_results, open(out_file, "wb"))


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

