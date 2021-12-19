cd ..
CUDA_VISIBLE_DEVICES=8 python eval_partial_motion_recovery.py \
--config ./clean_configs/clean_no_aug_len_64_set_1_val200_missing_lower_body_1.yaml \
--test_model /glab2/data/Users/jiaman/adobe/github/motion_prior/clean_outputs/clean_no_aug_len_64_set_1/checkpoints/gen_00250000.pt \
--final_motion_completion \
--trajectory_config ./clean_configs/clean_trajectory_pred_model_5s_set_3.yaml \
--trajectory_test_model ./clean_outputs/clean_trajectory_pred_model_5s_set_3/checkpoints/gen_00250000.pt
