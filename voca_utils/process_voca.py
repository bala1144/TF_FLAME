import glob
import os
import pickle

import numpy as np
import torch
from voca_utils.voca_data_cfg import get_default_config
from voca_utils.voca_data import DataHandler


class Data_binder():
    """
    Pipeline to create a bundled dataset pkl files for the ARD_ZDF dataset
    """
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        params = get_default_config()
        self.default_loaded_data = DataHandler(params, dataset_path=dataset_root)

    def _bind(self, process_seq, process_subjects, num_audio_feat=1):

        max_length = 0
        min_length = 100000
        data_dict = {}
        counter = 0
        for i, subj_name in enumerate(process_subjects):
            # print()
            print("process_subjects", process_subjects)
            # print("self.default_loaded_data.data2array_verts[subj_name]", self.default_loaded_data.data2array_verts[subj_name])
            # print()
            for j, sentence_name in enumerate(process_seq):
                seq_name = subj_name + "_" + sentence_name
                print(seq_name)
                exprs = self.default_loaded_data.data2array_verts[subj_name].get(sentence_name, None)

                if exprs is None:
                    print("Expressions not found")
                    continue

                expr_idxs = list(self.default_loaded_data.data2array_verts[subj_name][sentence_name].values())
                expression = self.default_loaded_data.face_vert_mmap[expr_idxs]
                expressen_len = len(expr_idxs)

                # get the processed audio
                process_audio_data = self.default_loaded_data.processed_audio[subj_name].get(sentence_name,None)
                if process_audio_data is None:
                    print("process_audio_data not found")
                    continue

                processed_audio = process_audio_data['audio']
                processed_audio_len = processed_audio.shape[0]
                processed_audio = processed_audio[:, :num_audio_feat].reshape(processed_audio_len, -1)
                processed_audio_sample_rate = self.default_loaded_data.processed_audio[subj_name][sentence_name]['sample_rate']

                raw_audio = self.default_loaded_data.raw_audio[subj_name][sentence_name]['audio']
                raw_audio_sample_rate = self.default_loaded_data.raw_audio[subj_name][sentence_name]['sample_rate']
                raw_audio_len = raw_audio.shape[0]

                if expressen_len != processed_audio_len:
                    print("exprssion len and audio len doestn match")

                seq_len = max(expressen_len, processed_audio_len)
                expression = expression[:seq_len]
                processed_audio = processed_audio[:seq_len]

                if max_length < seq_len:
                    max_length = seq_len
                if seq_len < min_length:
                    min_length = seq_len


                # create the sample dict
                data_dict[seq_name]={"seq_name":seq_name,
                                    "exprs":expression,
                                    "audio_feat":processed_audio,
                                    "processed_audio_sample_rate":processed_audio_sample_rate,
                                    "raw_audio":raw_audio,
                                    "raw_audio_sample_rate": raw_audio_sample_rate,
                                    "seq_len":seq_len
                                    }
                counter = counter+1

        print("NUmber of seqs", counter)
        data_dict['max_len'] = max_length
        data_dict['min_len'] = min_length
        return data_dict

    def run_bind(self, dataset='test', num_audio_feat=1, suffix=""):

        if dataset == "test":
            process_seq = self.default_loaded_data.testing_sequences
            process_subjects = self.default_loaded_data.testing_subjects
        elif dataset == "train":
            process_seq = self.default_loaded_data.training_sequences
            process_subjects = self.default_loaded_data.training_subjects
        elif dataset == "val":
            process_seq = self.default_loaded_data.validation_sequences
            process_subjects = self.default_loaded_data.validation_subjects
        else:
            raise("Invalid input set")

        data_dict = self._bind(process_seq, process_subjects, num_audio_feat)
        num_seq_in_dict = len(data_dict)-2
        # # store the datadict
        out_file = os.path.join(self.dataset_root, dataset+suffix+"_ns%s.pkl"%num_seq_in_dict)
        print('Writinfg pickle file', out_file)
        pickle.dump(data_dict, open(out_file, "wb"))
        print('Done writing')
        print()

class Data_generate_subseqs():

    def __init__(self):
        pass

    def split_data_given_idxs(self, key, seq, subset_idxs):
        seq_name, exprs, audio_feat, processed_audio_sample_rate, raw_audio, raw_audio_sample_rate, org_len = seq.values()

        data_splits = []
        data_split_name = []
        # data_dict[seq_name] = {"seq_name": seq_name,
        #                        "exprs": expression,
        #                        "audio_feat": processed_audio,
        #                        "processed_audio_sample_rate": processed_audio_sample_rate,
        #                        "raw_audio": raw_audio,
        #                        "raw_audio_sample_rate": raw_audio_sample_rate,
        #                        "seq_len": seq_len
        for spilt in subset_idxs:
            split_seq = {"seq_name": seq_name,
                         "exprs": exprs[spilt],
                         "audio_feat": audio_feat[spilt],
                         "raw_audio": raw_audio,
                         "raw_audio_sample_rate": raw_audio_sample_rate,
                         "seq_len": len(spilt),
                         "org_len": org_len
                         }
            # create the spilt name
            split_name = key + "_%03d" % spilt[0]
            # print("split_name", split_name)

            data_splits.append(split_seq)
            data_split_name.append(split_name)
        return data_split_name, data_splits

    def gen_subsplit_idx(self, seq_len, max_seq_len, num_seq_samples=30):
        all_idxs = np.arange(seq_len)
        possible_idxs = np.arange(0, seq_len - max_seq_len)
        idxs = np.unique(np.random.choice(possible_idxs, num_seq_samples))

        subsets = []
        for idx in idxs:
            subsets.append(all_idxs[idx:idx + max_seq_len])

        return subsets

    def add_split_to_datadict_from_list(self, dataset_dict, data_split_name, data_split):

        for key, val in zip(data_split_name, data_split):
            print(key, val["seq_name"])
            dataset_dict[key] = val
        return dataset_dict

    def split_seq_to_subseq(self, dataset_path, dataset_file, max_seq_len=256):
        """
        Function to split every simple sequence into multiples of max seq len, which is given as input
        """
        from taming.data.NVP_multi_seq import load_dataset_file
        samples = load_dataset_file(os.path.join(dataset_path, dataset_file))

        dataset_dict = {}
        for i, key in enumerate(samples):
            if "len" in key :
                continue
            print('Current Seq', key)
            seq = samples[key]
            seq_len = seq['seq_len']
            num_samples = (seq_len // max_seq_len) * 2
            subset_idxs = self.gen_subsplit_idx(seq_len, max_seq_len=max_seq_len, num_seq_samples=num_samples)
            data_split_name, data_split = self.split_data_given_idxs(key, seq, subset_idxs)
            self.add_split_to_datadict_from_list(dataset_dict, data_split_name, data_split)

        dataset_dict = add_min_max_to_samples(dataset_dict)
        print("max dataset_dict", len(dataset_dict)-2)
        dataset_file = dataset_file[:dataset_file.find(".pkl")]
        out_file = os.path.join(dataset_path, dataset_file + "_max_seq_len_np%s.pkl" % max_seq_len)
        store_file(out_file, dataset_dict)

