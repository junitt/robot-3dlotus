data_dir=data
seed=0
microstep_data_dir=$data_dir/gembench/train_dataset/microsteps/seed${seed}
keystep_data_dir=$data_dir/gembench/train_dataset/keysteps_bbox/seed${seed}
keystep_pcd_dir=$data_dir/gembench/train_dataset/keysteps_bbox_pcd_sam2/seed${seed}

num_episodes=100

python preprocess/gen_simple_policy_data.py \
    --input_dir ${keystep_data_dir} \
    --output_dir ${keystep_pcd_dir}