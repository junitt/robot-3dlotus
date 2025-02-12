expr_dir=data/experiments/gembench/3dlotusplus/v1
ckpt_step=140000
# test: with groundtruth task planner and groundtruth object grounding
#only evaluate l4 task record_video保存video
    # --record_video \
    # --video_dir data/video/seed${seed}\
for seed in {200..600..100}
do
for split in mytest
do
CUDA_VISIBLE_DEVICES=5 python genrobo3d/evaluation/eval_robot_pipeline_server.py \
    --full_gt \
    --record_video \
    --video_dir data/video/${split}/seed${seed}\
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