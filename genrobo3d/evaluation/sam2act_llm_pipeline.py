import torch
from easydict import EasyDict
import numpy as np
import copy,os,json
from genrobo3d.configs.default import get_config as get_model_config
from genrobo3d.evaluation.robot_pipeline import RobotPipeline
from genrobo3d.vlm_models.llm_task_planner import LlamaTaskPlanner
from genrobo3d.evaluation.eval_simple_policy import Actioner
from genrobo3d.evaluation.robot_pipeline_gt import GroundtruthTaskPlanner
from genrobo3d.evaluation.sam2act_pipeline import load_agent

from genrobo3d.utils.robot_box import RobotBox
from genrobo3d.utils.rvt_clip_preprocess import get_clip_model,get_embed
from genrobo3d.configs.rlbench.constants import get_robot_workspace
from genrobo3d.utils.point_cloud import voxelize_pcd
from genrobo3d.evaluation.common import parse_code
from genrobo3d.evaluation.sam2act_vlm_pipeline import Sam2RobotPipeline as Sam2VLMRobotPipeline
from genrobo3d.utils.rvt_util import instr_trans

class Sam2Actioner(Actioner):
    def __init__(self, exp_cfg=None,mvt_cfg=None, config = None) -> None:
        self.vlm_pipeline = Sam2VLMRobotPipeline(exp_cfg=exp_cfg,mvt_cfg=mvt_cfg, config=config)
        self.cache = None

    @torch.no_grad()#未修改
    def predict(self, taskvar, episode_id, step_id, instruction, obs_state_dict):
        task_str, variation = taskvar.split('+')
        out = self.vlm_pipeline.predict(task_str=task_str, variation=variation, step_id=step_id, obs_state_dict=obs_state_dict, episode_id=episode_id, instructions=[instruction], cache=self.cache)
        self.cache = out['cache']
        return out['action']