model_name=sam2act_voxel_unicolor_withobj_aug_10240
expr_dir=data/experiments/gembench/3dlotusplus/$model_name
ckpt_step=30000
# --record_video \
    # --video_dir data/video/$model_name/${split}/seed${seed}\
    # --not_include_robot_cameras \

for seed in {200..600..100}
do
for split in mytest
do
CUDA_VISIBLE_DEVICES=2,3 python genrobo3d/evaluation/eval_robot_pipeline_server.py \
    --eval_model_type sam2act \
    --max_steps 10\
    --record_video \
    --video_dir data/video/$model_name/${split}/seed${seed}\
    --not_include_robot_cameras \
    --pipeline_config_file genrobo3d/configs/rlbench/robot_pipeline.yaml \
    --mp_expr_dir ${expr_dir} \
    --mp_ckpt_step ${ckpt_step} \
    --num_workers 1 \
    --taskvar_file assets/taskvars_${split}.json \
    --seed ${seed} --num_demos 20 \
    --microstep_data_dir data/gembench/test_dataset/microsteps/seed${seed} \
    --pc_label_type coarse --run_action_step 1 
done
done