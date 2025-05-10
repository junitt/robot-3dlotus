import os
import argparse

import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import random
import yaml
from easydict import EasyDict
from flask import Flask, request
from genrobo3d.train.utils.rvt_utils import load_cfgs
from genrobo3d.evaluation.sam2act_llm_pipeline import Sam2Actioner


def main(args):
    app = Flask(__name__)
    #pipeline_config_file
    #llm_master_port
    #gt_og_label_file
    #run_action_step
    #mp_expr_dir
    #mp_ckpt_step
    #exp_cfg_path
    #exp_cfg_opts:""
    #mvt_cfg_path
    #mvt_cfg_opts:""

    with open(args.pipeline_config_file, 'r') as f:
        pipeline_config = yaml.safe_load(f)
    pipeline_config = EasyDict(pipeline_config)

    pipeline_config.llm_planner.use_groundtruth = False
    if args.llm_master_port is not None:
        pipeline_config.llm_planner.master_port = args.llm_master_port

    if args.gt_og_label_file is not None:
        pipeline_config.object_grounding.gt_label_file = args.gt_og_label_file
    pipeline_config.motion_planner.run_action_step = args.run_action_step

    if args.mp_expr_dir is None:
        args.mp_expr_dir = pipeline_config.motion_planner.expr_dir
    if args.mp_ckpt_step is None:
        args.mp_ckpt_step = pipeline_config.motion_planner.ckpt_step
    mp_checkpoint_file = os.path.join(
        args.mp_expr_dir, 'ckpts', f'model_{args.mp_ckpt_step}.pth'
    )
    args.exp_cfg_path = os.path.join(
        args.mp_expr_dir, 'logs', 'exp_cfg.yaml'
    )
    args.mvt_cfg_path = os.path.join(
        args.mp_expr_dir, 'logs', 'mvt_cfg.yaml'
    )
    if not os.path.exists(mp_checkpoint_file):
        print(mp_checkpoint_file, 'not exists')
        return
    
    pipeline_config.motion_planner.expr_dir = args.mp_expr_dir
    pipeline_config.motion_planner.ckpt_step = args.mp_ckpt_step
    pipeline_config.motion_planner.checkpoint = mp_checkpoint_file
    pipeline_config.motion_planner.config_file = os.path.join(
        args.mp_expr_dir, 'logs', 'training_config.yaml'
    )
    pipeline_config.motion_planner.save_obs_outs = False
    exp_cfg,mvt_cfg = load_cfgs(args)
    actioner = Sam2Actioner(exp_cfg,mvt_cfg,pipeline_config)

    @app.route('/predict', methods=['POST'])
    def predict():
        '''
        batch is a dict containing:
            taskvar: str, 'task+variation'
            episode_id: int
            step_id: int, [0, 25]
            instruction: str
            obs_state_dict: observations from genrobo3d.rlbench.environments.RLBenchEnv 
        '''
        data = request.data
        batch = msgpack_numpy.unpackb(data)
        print(batch.keys())
        action = actioner.predict(**batch)

        action = msgpack_numpy.packb(action)
        return action
    
    app.run(host=args.ip, port=args.port, debug=args.debug)


if __name__ == "__main__":
    #pipeline_config_file
    #llm_master_port
    #gt_og_label_file
    #run_action_step
    #mp_expr_dir
    #mp_ckpt_step
    #exp_cfg_path
    #exp_cfg_opts:""
    #mvt_cfg_path
    #mvt_cfg_opts:""
    parser = argparse.ArgumentParser(description='Actioner server')
    parser.add_argument('--ip', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=13000)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--pipeline_config_file', type=str, default="genrobo3d/configs/rlbench/robot_pipeline.yaml")
    parser.add_argument('--llm_master_port', type=int, default=15522+random.randint(0,1000))
    parser.add_argument('--gt_og_label_file',type=str, default='assets/taskvars_train_target_label.json')
    parser.add_argument('--run_action_step',type=int, default=1)
    parser.add_argument('--mp_expr_dir',type=str, default="data/experiments/gembench/3dlotusplus/augbutton_imgidx")
    parser.add_argument('--mp_ckpt_step',type=int, default=80000)
    parser.add_argument('--exp_cfg_path',type=str, default="")
    parser.add_argument('--exp_cfg_opts',type=str, default="")
    parser.add_argument('--mvt_cfg_path',type=str, default="")
    parser.add_argument('--mvt_cfg_opts',type=str, default="")
    args = parser.parse_args()
    main(args)
