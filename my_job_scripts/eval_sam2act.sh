expr_dir=data/experiments/gembench/3dlotusplus/sam2act_test
ckpt_step=50000
# test: with groundtruth task planner and groundtruth object grounding
#only evaluate l4 task record_video保存video
    # --record_video \
    # --video_dir data/video/${split}/seed${seed}\
    # --not_include_robot_cameras \
for seed in {200..600..100}
do
for split in train
do
CUDA_VISIBLE_DEVICES=0 python genrobo3d/evaluation/eval_robot_pipeline_server.py \
    --full_gt \
    --record_video \
    --eval_model_type sam2act \
    --exp_cfg_path configs/sam2act.yaml \
    --mvt_cfg_path mvt/configs/sam2act_gembench.yaml \
    --video_dir data/video/${split}/seed${seed}\
    --not_include_robot_cameras \
    --max_steps 15\
    --pipeline_config_file genrobo3d/configs/rlbench/robot_pipeline_gt.yaml \
    --mp_expr_dir ${expr_dir} \
    --mp_ckpt_step ${ckpt_step} \
    --num_workers 1 \
    --taskvar_file assets/taskvars_${split}.json \
    --gt_og_label_file assets/taskvars_target_label_zrange.json \
    --seed ${seed} --num_demos 20 \
    --microstep_data_dir data/gembench/test_dataset/microsteps/seed${seed} \
    --pc_label_type coarse --run_action_step 1
done
done