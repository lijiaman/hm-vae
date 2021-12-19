cd ..
CUDA_VISIBLE_DEVICES=4 python eval_partial_motion_recovery.py \
--config ./clean_configs/final_two_hier_model_no_aug_len_64_set_11_test_interpolation_3.yaml \
--test_model ./clean_outputs/final_two_hier_model_no_aug_len_64_set_11/checkpoints/gen_00250000.pt \
--try_interpolation_w_trajectory_single_window \
--trajectory_config ./clean_configs/clean_trajectory_pred_model_5s_set_3.yaml \
--trajectory_test_model ./clean_outputs/clean_trajectory_pred_model_5s_set_3/checkpoints/gen_00250000.pt
