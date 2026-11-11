#! /bin/bash

#define environment variables 
export nnUNet_raw="/home/wirrbel/MN_8/2026-06-01_nnunet_test/training_data/wfa/nnUNet_raw"
export nnUNet_results="/home/wirrbel/MN_8/2026-06-01_nnunet_test/training_data/wfa/nnUNet_results"
export nnUNet_preprocessed="/home/wirrbel/MN_8/2026-06-01_nnunet_test/training_data/wfa/nnUNet_preprocessed"

#run training command 
nnUNetv2_plan_and_preprocess -d 021 -c 3d_fullres -pl nnUNetPlannerResEncL --verify_dataset_integrity

#CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 021 3d_fullres 0 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_50epochs --npz 
#CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 021 3d_fullres 1 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_50epochs --npz
#CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 021 3d_fullres 2 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_50epochs --npz
#CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 021 3d_fullres 3 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_50epochs --npz
#CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 021 3d_fullres 4 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_50epochs --npz

#postprocess
#nnUNetv2_find_best_configuration -d 21 -c 3d_fullres -p nnUNetPlannerResEncL -tr nnUNetTrainer_50epochs 