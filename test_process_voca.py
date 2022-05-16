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
import trimesh
from voca_utils.voca_data_cfg import get_default_config
import torch

def test():
    dataset_path = os.path.join(os.getenv('HOME'), "projects/dataset/voca")
    binder = Data_binder(dataset_path)
    # binder.run_bind("test", num_audio_feat=4)
    binder.run_bind("train", num_audio_feat=4, suffix="FaceTalk_170725_00137_TA")
    # binder.run_bind("val", num_audio_feat=4)
    # binder.run_bind("val")

def save_sample_voca_meshes():
    dataset_path = os.path.join(os.getenv('HOME'), "projects/dataset/voca")
    face_verts_mmaps_path = os.path.join(dataset_path,"data_verts.npy")
    face_vert_mmap = np.load(face_verts_mmaps_path, mmap_mode='r')
    templates_data = pickle.load(open(os.path.join(dataset_path, "templates.pkl"), 'rb'), encoding="latin1")

    target_mesh_path = os.path.join(os.getenv('HOME'), "projects/TF_FLAME", "data/registered_mesh.ply")
    target_mesh = Mesh(filename=target_mesh_path)
    flames_faces = target_mesh.f

    outfile_path = os.path.join(dataset_path, "sample_mesh_ply")
    os.makedirs(outfile_path, exist_ok=True)


    fitted_outfile_path = os.path.join(dataset_path, "fitted_sample_mesh_ply_expr_e-3")
    os.makedirs(fitted_outfile_path, exist_ok=True)

    num_verts = face_vert_mmap.shape[0]
    import random
    sample_idx = random.sample(range(0, num_verts), 20)

    for i in sample_idx:
        print("Writing frames", i)
        v = face_vert_mmap[i]
        mesh = Mesh(v, flames_faces)
        out_mesh = os.path.join(outfile_path, "mesh_%07d.ply" % i)
        mesh.write_ply(out_mesh)

        result_mesh, pose, rot, trans, shape, exprs = fit_3D_mesh(mesh.v, mesh.f)
        out_mesh = os.path.join(fitted_outfile_path, "mesh_fitted%07d.ply" % i)
        result_mesh.write_ply(out_mesh)
        print("pose", pose)
        print("rot", rot)
        print("trans", trans)
        print("shape", shape)
        print("exprs", exprs)

    # # template mesh
    # template_outfile_path = os.path.join(dataset_path, "template_mesh_ply")
    # os.makedirs(template_outfile_path, exist_ok=True)
    # for key, v in templates_data.items():
    #     print("Writing frames", key)
    #     mesh = Mesh(v, flames_faces)
    #     out_mesh = os.path.join(template_outfile_path, key+".ply")
    #     mesh.write_ply(out_mesh)
    #
    #     result_mesh, pose, rot, trans, shape, exprs = fit_3D_mesh(mesh.v, mesh.f)
    #     out_mesh = os.path.join(template_outfile_path, key + ".ply")
    #     mesh.write_ply(out_mesh)

def sequence_specific_fitting():

    dataset_path = os.path.join(os.getenv('HOME'), "projects/dataset/voca")
    outfile_path = os.path.join(dataset_path, "trainFaceTalk_170904_00128_TA_ns40_fitted_mesh")
    os.makedirs(outfile_path, exist_ok=True)
    font = cv2.FONT_HERSHEY_SIMPLEX


    subject1 = os.path.join(dataset_path, "trainFaceTalk_170904_00128_TA_ns40.pkl")
    # subject2 = "/Users/bthambiraja/projects/dataset/voca/trainFaceTalk_170728_03272_TA_ns40.pkl"
    loaded_data = pickle.load(open(subject1, 'rb'), encoding="latin1")

    target_mesh_path = os.path.join(os.getenv('HOME'), "projects/TF_FLAME", "data/registered_mesh.ply")
    template_mesh = trimesh.load_mesh(target_mesh_path, process=False)
    flames_faces = template_mesh.faces
    # mesh_render = flame_render(width=512, height=512)
    mesh_render = Facerender()


    fitted_results = {}
    for seq, seq_dict in loaded_data.items():

        if 'len' in seq:
            continue
        # running
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

def split_based_on_seq():
    dataset_path = os.path.join(os.getenv('HOME'), "projects/dataset/voca")
    # binder = Data_binder(dataset_path)clear

    params = get_default_config()
    seqs_list = params["sequence_for_training"].split()
    subjects = params["all_subjects"].split()

    for sub in subjects:
        for seq in seqs_list:
            # print("Running on seq", seq)
            print(sub+"_"+seq)

# def extract_all_seq():
#     dataset_path = os.path.join(os.getenv('HOME'), "projects/dataset/voca")
#     binder = Data_binder(dataset_path)
#     params = get_default_config()
#     seqs_list = params["sequence_for_training"].split()
#     subjects = params["all_subjects"].split()
#     binder.run_bind_seqwise(seqs_list, subjects)


def combine_loaded_sequence(loaded_data):
    result_vertices_cum = []
    pose_cum = []
    rot_cum = []
    trans_cum = []
    shape_cum = []
    exprs_cum = []
    for subj_seq, params in loaded_data.items():
        for frame in params:
            result_vertices, pose, rot, trans, shape, exprs = frame
            result_vertices_cum.append(result_vertices)
            pose_cum.append(pose)
            rot_cum.append(rot)
            trans_cum.append(trans)
            shape_cum.append(shape)
            exprs_cum.append(exprs)

    # accumulate the results
    result_vertices_cum = np.asarray(result_vertices_cum)
    pose_cum = np.asarray(pose_cum)
    rot_cum = np.asarray(rot_cum)
    trans_cum = np.asarray(trans_cum)
    shape_cum = np.asarray(shape_cum)
    exprs_cum = np.asarray(exprs_cum)

    # make them as dict
    data_dict = {}
    # data_dict["optim_mesh"] = result_vertices_cum
    data_dict["pose"] = pose_cum.squeeze(axis=1)
    data_dict["rot"] = rot_cum.squeeze(axis=1)
    data_dict["trans"] = trans_cum.squeeze(axis=1)
    data_dict["shape"] = shape_cum.squeeze(axis=1)
    data_dict["exprs"] = exprs_cum.squeeze(axis=1)

    return data_dict

def accumulate_dataset_results():
    dataset_path = os.path.join(os.getenv('HOME'), "projects/dataset/voca", "subj_seq_fitting_results_w_mean_shape")

    params = get_default_config()
    seqs_list = params["sequence_for_training"].split()
    subjects = params["all_subjects"].split()

    accumulated_dict = {}
    i = 0
    for sub in subjects:
        for seq in seqs_list:
            # print("Running on seq", seq)
            print(sub+"_"+seq)
            current_dataset_file = os.path.join(dataset_path, sub, sub+"_"+seq + ".pkl")

            if os.path.isfile(current_dataset_file):
                loaded_data = pickle.load(open(current_dataset_file, 'rb'), encoding="latin1")
                data_dict = combine_loaded_sequence(loaded_data)
                data_dict["seq_name"] = sub+"_"+seq

                # add to accumulated dict
                accumulated_dict[sub+"_"+seq] = data_dict
                print("curren file %d", i)
                i+=1
            else:
                print("File doesnt exist", current_dataset_file)


    accumulated_extracted_file = os.path.join(dataset_path, "accumulated_extracted_nomesh_ns%d.pkl"%len(accumulated_dict))
    # accumulated_extracted_file = os.path.join(dataset_path, "accumulated_extracted_ns%d.pkl"%len(accumulated_dict))
    pickle.dump(accumulated_dict, open(accumulated_extracted_file, "wb"))
    print("File out", accumulated_extracted_file)
    print("Number of samples in the dict", len(accumulated_dict))



def get_flame_face_given_expression(flame_model, exprs):
    exprs = torch.from_numpy(exprs).float()
    if torch.cuda.is_available():
        exprs = exprs.cuda()

    vertices = flame_model.morph(exprs).cpu().numpy()
    return vertices


def compute_identity_from_results():
    dataset_path = os.path.join(os.getenv('HOME'), "projects/dataset/voca", "subj_seq_fitting_results_w_mean_shape")
    params = get_default_config()
    seqs_list = params["sequence_for_training"].split()
    subjects = params["all_subjects"].split()

    accumulated_extracted_file = os.path.join(dataset_path, "accumulated_extracted_nomesh_ns473.pkl")
    loaded_data = pickle.load(open(accumulated_extracted_file, 'rb'), encoding="latin1")

    subj_mean_shape = {}
    for sub in subjects:
        print("Running on subject", sub)
        current_subject_seq_means = []
        for seq in seqs_list:
            seq_name = sub+"_"+seq
            try:
                current_seq_mean = loaded_data[seq_name]["shape"].mean(axis=0)
            except:
                print(seq_name, "doesnt exist")
            current_subject_seq_means.append(current_seq_mean)

        # find the cumulative mean
        subj_mean_shape[sub] = np.mean(current_subject_seq_means, axis=0)

    # subjwise mean shape
    subjectwise_mean_shape_file = os.path.join(dataset_path, "subjectwise_mean_shape.pkl")
    pickle.dump(subj_mean_shape, open(subjectwise_mean_shape_file, "wb"))
    print("File out", subjectwise_mean_shape_file)


if __name__ == "__main__":
    # test()
    # save_sample_voca_meshes()
    # sequence_specific_fitting()
    # fitting_voca()
    # split_based_on_seq()
    # compute_identity_from_results()
    accumulate_dataset_results()