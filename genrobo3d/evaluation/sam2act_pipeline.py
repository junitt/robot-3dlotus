from genrobo3d.evaluation.robot_pipeline_gt import GroundtruthRobotPipeline,GroundtruthVision,GroundtruthTaskPlanner
import os
from genrobo3d.models.sam2act_agent import SAM2Act_Agent2 as SAM2Act_Agent
from genrobo3d.train.utils.rvt_utils import get_model_size,load_cfgs
from genrobo3d.configs.default import get_config as get_model_config
import torch
from genrobo3d.utils.rvt_clip_preprocess import get_clip_model,get_embed

from sam2act.utils.rvt_utils import load_agent_only_model as load_agent_state
import sam2act.mvt.mvt_sam2 as mvt_sam2
from sam2act.utils.peract_utils import (
    IMAGE_SIZE,
)
from genrobo3d.configs.rlbench.constants import SCENE_BOUNDS
def load_agent(
    model_path=None,
    exp_cfg=None,
    mvt_cfg=None,
    eval_log_dir="",
    device=0,
    use_input_place_with_mean=False,
):

    assert model_path is not None
    assert mvt_cfg is not None 
    assert exp_cfg is not None
    # load exp_cfg

    # NOTE: to not use place_with_mean in evaluation
    # needed for rvt-1 but not rvt-2
    if not use_input_place_with_mean:
        # for backward compatibility
        old_place_with_mean = exp_cfg.rvt.place_with_mean
        exp_cfg.rvt.place_with_mean = True

    exp_cfg.freeze()
    # create agent
    if exp_cfg.agent == "our":

        mvt_cfg.freeze()

        # for rvt-2 we do not change place_with_mean regardless of the arg
        # done this way to ensure backward compatibility and allow the
        # flexibility for rvt-1
        if mvt_cfg.stage_two:
            exp_cfg.defrost()
            exp_cfg.rvt.place_with_mean = old_place_with_mean
            exp_cfg.freeze()

        sam2act = mvt_sam2.MVT_SAM2(
            renderer_device=device,
            rank=0,
            **mvt_cfg,
        )

        get_model_size(sam2act)

        agent = SAM2Act_Agent(
            use_sem=exp_cfg.sam2_use_sem,
            network=sam2act.to(device),
            image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
            add_lang=mvt_cfg.add_lang,
            stage_two=mvt_cfg.stage_two,
            rot_ver=mvt_cfg.rot_ver,
            scene_bounds=SCENE_BOUNDS,
            cameras=None,# not used
            log_dir=f"{eval_log_dir}/eval_run",
            **exp_cfg.peract,
            **exp_cfg.rvt,
        )
    else:
        raise NotImplementedError

    agent.build(training=False, device=device)
    load_agent_state(model_path, agent)
    agent.eval()
    return agent

class Sam2RobotPipeline(GroundtruthRobotPipeline):
    def __init__(self,exp_cfg=None,mvt_cfg=None, config = None):
        #config for original evaluation
        self.exp_cfg = exp_cfg
        self.mvt_cfg = mvt_cfg
        self.model_class = "sam2act"
        self.config = config
        self.device = "cuda:0"

        # build LLM high-level planner
        llm_config = config.llm_planner
        self.llm_planner = GroundtruthTaskPlanner(llm_config.gt_plan_file)
        
        mp_expr_dir = config.motion_planner.expr_dir
        mp_config_file = config.motion_planner.config_file
        mp_config = get_model_config(mp_config_file)
        data_cfg = mp_config.TRAIN_DATASET
        self.instr_include_objects = data_cfg.get('instr_include_objects', False)
        self.vlm_pipeline = GroundtruthVision(
            self.config.object_grounding.gt_label_file,
            num_points=data_cfg.num_points, voxel_size=mp_config.MODEL.action_config.voxel_size, 
            same_npoints_per_example=data_cfg.same_npoints_per_example, rm_robot=data_cfg.rm_robot,
            xyz_shift=data_cfg.xyz_shift, xyz_norm=data_cfg.xyz_norm, use_height=data_cfg.use_height,
            pc_label_type=data_cfg.pc_label_type if config.motion_planner.pc_label_type is None else config.motion_planner.pc_label_type, use_color=data_cfg.get('use_color', False),
            model_class=self.model_class,rm_table=data_cfg.rm_table
        )

        # build motion planner
        # self.clip_model = OpenClipEncoder(device=self.device) # to encode action/object texts
        self.motion_planner = self.build_motion_planner(config.motion_planner, device=self.device)
        print(f"test model {self.model_class}")
        assert self.model_class in ['sam2act']

        self.clip_model = get_clip_model()

        # caches
        self.set_system_caches()
        self._ori_gripper_pose={}#{taskvar_episode_id:action}

    def build_motion_planner(self, mp_config, device):
        original_path = mp_config.checkpoint
        agent = load_agent(
            model_path=original_path,
            exp_cfg=self.exp_cfg,
            mvt_cfg=self.mvt_cfg,
            eval_log_dir="runs",
            device=self.device,
        )
        return agent