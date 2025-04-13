import os
import json
import numpy as np

from tqdm import tqdm
from genrobo3d.utils.rvt_clip_preprocess import get_clip_model,get_embed
from genrobo3d.vlm_models.clip_encoder import ClipEncoder, OpenClipEncoder
from genrobo3d.utils.rvt_util import replace_color
from genrobo3d.configs.rlbench.constants import COLORS
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name', default='clip', choices=['openclip', 'clip'])
    parser.add_argument('--include_objects', default=False, action='store_true')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.include_objects:
        output_file = os.path.join(args.output_dir, f'instr_embeds_aug_sam2.npy')
    else:
        output_file = os.path.join(args.output_dir, f'instr_embeds_aug_sam2_no_obj.npy')
    
    taskvars_target_labels = json.load(open('assets/taskvars_target_label_zrange.json'))
    color_dict=json.load(open("assets/train_aug_color.json"))
    colors = [color[0] for color in COLORS]
    action_names = set()
    for taskvar in taskvars_target_labels.keys():
        task, variation = taskvar.split('+')
        target_labels = taskvars_target_labels[taskvar]
        aug_txt = taskvar in color_dict.keys()
        
        if aug_txt:
            aug_label = color_dict[taskvar]['aug_label']
            except_color_lst = color_dict[taskvar]['distractor_colors']
            # print(taskvar)
            
        for target_label in target_labels:
            action_name = target_label['action']
            if 'object' in target_label:
                action_name = f"{action_name} {target_label['object']['name']}"
            if 'target' in target_label:
                action_name = f"{action_name} to {target_label['target']['name']}"
            action_names.add(action_name)
            
            if task == "stack_blocks" and target_label['action'] == "move grasped object" and \
            target_label["target"]["name"]!="green square":
                    aug_label = "target"
            if aug_txt and (aug_label in target_label):
                for target_color in colors:
                    aug_instr = replace_color(aug_label,target_label,target_color,except_color_lst)
                    action_names.add(aug_instr)
    print(len(action_names), action_names)

    
    action_embeds = {}
    clip_model=get_clip_model()
    for query in action_names:
        txt_embed = get_embed(clip_model,query)
        action_embeds[query] = txt_embed

    np.save(output_file, action_embeds)

 
if __name__ == '__main__':
    main()
