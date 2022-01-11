'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.
More information about VOCA is available at http://voca.is.tue.mpg.de.
For comments or questions, please email us at voca@tue.mpg.de
'''

import os
import configparser

def set_default_paramters(config):

    config.add_section('Data Setup')
    # config.set('Data Setup', 'subject_for_training',
               # "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA "
               # "FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    config.set('Data Setup', 'subject_for_training',
               "FaceTalk_170725_00137_TA")
    config.set('Data Setup', 'sequence_for_training',
                "sentence01 sentence02 sentence03 sentence04 sentence05 sentence06 sentence07 sentence08 sentence09 sentence10 "
                "sentence11 sentence12 sentence13 sentence14 sentence15 sentence16 sentence17 sentence18 sentence19 sentence20 "
                "sentence21 sentence22 sentence23 sentence24 sentence25 sentence26 sentence27 sentence28 sentence29 sentence30 "
                "sentence31 sentence32 sentence33 sentence34 sentence35 sentence36 sentence37 sentence38 sentence39 sentence40")
    config.set('Data Setup', 'subject_for_validation', "FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA")
    config.set('Data Setup', 'sequence_for_validation',
                "sentence21 senteence22 sentence23 sentence24 sentence25 sentence26 sentence27 sentence28 sentence29 sentence30 "
                "sentence31 sentence32 sentence33 sentence34 sentence35 sentence36 sentence37 sentence38 sentence39 sentence40")
    config.set('Data Setup', 'subject_for_testing', "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA")
    config.set('Data Setup', 'sequence_for_testing',
                "sentence21 sentence22 sentence23 sentence24 sentence25 sentence26 sentence27 sentence28 sentence29 sentence30 "
                "sentence31 sentence32 sentence33 sentence34 sentence35 sentence36 sentence37 sentence38 sentence39 sentence40")
    config.set('Data Setup', 'num_consecutive_frames', '10')  # 2

    config.set('Data Setup', 'all_subjects',
            "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA "
            "FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA "
            "FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA")





def create_default_config(fname):
    config = configparser.ConfigParser()
    set_default_paramters(config)

    with open(fname, 'w') as configfile:
        config.write(configfile)
        configfile.close()

def get_default_config():

    config = configparser.ConfigParser()
    set_default_paramters(config)
    config_parms = {}
    config_parms['subject_for_training'] = config.get('Data Setup', 'subject_for_training')
    config_parms['sequence_for_training'] = config.get('Data Setup', 'sequence_for_training')
    config_parms['subject_for_validation'] = config.get('Data Setup', 'subject_for_validation')
    config_parms['sequence_for_validation'] = config.get('Data Setup', 'sequence_for_validation')
    config_parms['subject_for_testing'] = config.get('Data Setup', 'subject_for_testing')
    config_parms['sequence_for_testing'] = config.get('Data Setup', 'sequence_for_testing')
    config_parms['num_consecutive_frames'] = config.get('Data Setup', 'num_consecutive_frames')
    config_parms['all_subjects'] = config.get('Data Setup', 'all_subjects')

    return config_parms