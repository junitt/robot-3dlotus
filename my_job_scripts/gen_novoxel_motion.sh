data_dir=/data5/lzy/data/gembench
seed=0
microstep_data_dir=$data_dir/gembench/train_dataset/microsteps/seed${seed}
keystep_data_dir=$data_dir/gembench/train_dataset/keysteps_bbox/seed${seed}
keystep_pcd_dir=data/gembench/train_dataset/keysteps_bbox_pcd/seed0/voxel1cm
motion_dir=$data_dir/motion_withtable_pcd/seed${seed}
num_episodes=100

# Convert to point clouds
python preprocess/gen_motion_planner_data.py \
 --old_keystep_pcd_dir ${keystep_pcd_dir} \
 --new_keystep_pcd_dir ${motion_dir}