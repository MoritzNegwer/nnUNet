#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 00:36:24 2026

@author: wirrbel
"""

import os
import torch

import sys
sys.path.append('/home/wirrbel/2026-10-27_nnunet_normalizer_test/nnUNet')

from nnunetv2.run.run_training import run_training
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprint_dataset, plan_experiment_dataset, preprocess_dataset
from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import nnUNetPlannerResEncL 

from nnunetv2.configuration import default_num_processes
from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.preprocessing.sampling_locations.extract_sampling_locations import (
    extract_sampling_locations_for_folder)
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.file_path_utilities import copy_file_if_newer
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets



#define environment variables 
os.environ["nnUNet_raw"]="/home/wirrbel/MN_8/2026-06-01_nnunet_test/training_data/wfa/nnUNet_raw"
os.environ["nnUNet_results"]="/home/wirrbel/MN_8/2026-06-01_nnunet_test/training_data/wfa/nnUNet_results"
os.environ["nnUNet_preprocessed"]="/home/wirrbel/MN_8/2026-06-01_nnunet_test/training_data/wfa/nnUNet_preprocessed"

#set up cuda variables 
os.environ["CUDA_VISIBLE_DEVICES"]="1"


'''
#run training command 
nnUNetv2_plan_and_preprocess -d 021 -c 3d_fullres -pl nnUNetPlannerResEncL --verify_dataset_integrity
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 021 3d_fullres 0 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_50epochs --npz 
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 021 3d_fullres 1 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_50epochs --npz
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 021 3d_fullres 2 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_50epochs --npz
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 021 3d_fullres 3 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_50epochs --npz
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 021 3d_fullres 4 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_50epochs --npz

#postprocess
nnUNetv2_find_best_configuration -d 21 -c 3d_fullres -p nnUNetPlannerResEncL -tr nnUNetTrainer_50epochs 

'''

def main():
    #preprocess
    
    extract_fingerprint_dataset(dataset_id = 21,
                                fingerprint_extractor_class = DatasetFingerprintExtractor,
                                num_processes = 16, check_dataset_integrity = True,
                                clean= True, verbose = True,
                                show_progress_bar= True)
    
    ret, plans_identifier = plan_experiment_dataset(dataset_id = 21,
                                experiment_planner_class = nnUNetPlannerResEncL,
                                gpu_memory_target_in_gb = None, preprocess_class_name = 'DefaultPreprocessor',
                                )
    
    
    preprocess_dataset(dataset_id= 21,
                   plans_identifier = plans_identifier,
                   configurations = ['3d_fullres'],
                   num_processes = [16],
                   verbose= False,
                   show_progress_bar= True)
    
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda')

    #run the training, one fold at a time 
    run_training("021", '3d_fullres', 0 , "nnUNetTrainer_50epochs", "nnUNetResEncUNetLPlans", None,
                     1, True, device=torch.device('cuda'))
    
    run_training("021", '3d_fullres', 1 , "nnUNetTrainer_50epochs", "nnUNetResEncUNetLPlans", None,
                     1, True, device=torch.device('cuda'))
    
    run_training("021", '3d_fullres', 2 , "nnUNetTrainer_50epochs", "nnUNetResEncUNetLPlans", None,
                     1, True, device=torch.device('cuda'))
    
    run_training("021", '3d_fullres', 3 , "nnUNetTrainer_50epochs", "nnUNetResEncUNetLPlans", None,
                     1, True, device=torch.device('cuda'))
    
    run_training("021", '3d_fullres', 4 , "nnUNetTrainer_50epochs", "nnUNetResEncUNetLPlans", None,
                     1, True, device=torch.device('cuda'))
    
if __name__ == "__main__":
    main()