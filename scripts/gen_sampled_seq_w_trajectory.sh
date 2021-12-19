cd ..
CUDA_VISIBLE_DEVICES=2 python eval_trajectory_pred.py \
--config ./clean_configs/clean_no_aug_len_64_set_1.yaml \
--test_model ./clean_outputs/clean_no_aug_len_64_set_1/checkpoints/gen_00250000.pt \
--trajectory_config ./clean_configs/ready_trajectory_model_set_1.yaml \
--trajectory_test_model ./clean_outputs/ready_trajectory_model_set_1/checkpoints/gen_00250000.pt \
--pred_trajectory_for_single_window
