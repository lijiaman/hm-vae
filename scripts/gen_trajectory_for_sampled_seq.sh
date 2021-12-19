cd ..
CUDA_VISIBLE_DEVICES=3 python eval_trajectory_pred.py \
--config ./clean_configs/clean_no_aug_len_64_set_1.yaml \
--test_model ./clean_outputs/clean_no_aug_len_64_set_1/checkpoints/gen_00250000.pt \
--trajectory_config ./clean_configs/ready_trajectory_model_set_1.yaml \
--trajectory_test_model ./clean_outputs/ready_trajectory_model_set_1/checkpoints/gen_00250000.pt \
--seq_generation_npy_folder /glab2/data/Users/jiaman/adobe/github/VIBE/mesh_vis_our_decoded_3dpw_eval_results/clean_len_8_vae_set_4
