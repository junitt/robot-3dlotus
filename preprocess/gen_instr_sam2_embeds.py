import os
import json
import numpy as np

from tqdm import tqdm
from genrobo3d.utils.rvt_clip_preprocess import get_clip_model,get_embed
from genrobo3d.vlm_models.clip_encoder import ClipEncoder, OpenClipEncoder

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name', default='clip', choices=['openclip', 'clip'])
    parser.add_argument('--include_objects', default=False, action='store_true')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.include_objects:
        output_file = os.path.join(args.output_dir, f'instr_embeds_sam2.npy')
    else:
        output_file = os.path.join(args.output_dir, f'instr_embeds_sam2_no_obj.npy')
    
    taskvars_target_labels = json.load(open('assets/taskvars_target_label_zrange.json'))
    action_names = set()
    for target_labels in taskvars_target_labels.values():
        for target_label in target_labels:
            action_name = target_label['action']
            if args.include_objects:
                if 'object' in target_label:
                    action_name = f"{action_name} {target_label['object']['name']}"
                if 'target' in target_label:
                    action_name = f"{action_name} to {target_label['target']['name']}"
            action_names.add(action_name)
    print(len(action_names), action_names)

    
    action_embeds = {}
    clip_model=get_clip_model()
    for query in action_names:
        txt_embed = get_embed(clip_model,query)
        action_embeds[query] = txt_embed

    np.save(output_file, action_embeds)

 
if __name__ == '__main__':
    main()
