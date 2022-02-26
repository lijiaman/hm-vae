cd ..
CUDA_VISIBLE_DEVICES=5 python eval_partial_motion_recovery.py \
--config ./configs/len_64_test_interpolation.yaml \
--test_model ./outputs/len64_no_aug_hm_vae/checkpoints/gen_00250000.pt \
--final_try_long_seq_interpolation \
--trajectory_config ./configs/trajectory_model.yaml \
--trajectory_test_model ./outputs/trajectory_model/checkpoints/gen_00250000.pt
