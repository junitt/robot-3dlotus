from genrobo3d.configs.rlbench.constants import COLORS
import copy
def instr_trans(instruction):
    #object 1, target 2, object+target 3
    #move_grasped_object possible for up down and out
    ori_instr = [("move grasped object",2),("push down",1),("push forward",3),("grasp",1)]
    action_name = None
    action_form_id = -1
    for keyword in ori_instr:
        if keyword[0] in instruction:
            action_name = keyword[0]
            action_form_id = keyword[1]
            break
    assert action_name is not None
    
    if action_form_id==1:
        action_name+=" blue object"
    elif action_form_id==2:
        if " to " in instruction:
            action_name+=" to yellow target"
        else:
            action_name = instruction
    elif action_form_id==3:
        action_name+=" blue object"
        if " to " in instruction:
            action_name+=" to yellow target"
    else:
        assert False
    return action_name

def unify_color(pc_labels,pc_fts,offsets=None,bs=1):
    #bs = len(batch['txt_lens'])
    # offsets=batch['offset'].cpu().numpy().tolist()
    # offsets=[0]+offsets

    # 0: obstacle, 1: robot, 2: object, 3: target
    # obstacle('white', (1.0, 1.0, 1.0))
    # robot('red', (1.0, 0.0, 0.0))
    # object('blue', (0.0, 0.0, 1.0))
    # target('yellow', (1.0, 1.0, 0.0))
    assert pc_labels.shape[0]==pc_fts.shape[0]
    colors=[(1.0, 1.0, 1.0),(1.0, 0.0, 0.0),(0.0, 0.0, 1.0),(1.0, 1.0, 0.0)]

    for i in range(len(pc_labels)):
        color = colors[pc_labels[i]]
        for idx in range(3):
            pc_fts[i,idx-3] = color[idx]/2
    if offsets is not None:
        #batch data
        img_feat=[pc_fts[:, -3:][offsets[i]:offsets[i+1]] for i in range(bs)]
    else:
        img_feat=pc_fts
    return img_feat

def replace_color(aug_label,target_label,target_color,exp_col_lst):
    assert aug_label in ['object','target']
    s = target_label[aug_label]['name']
    target_label = copy.deepcopy(target_label)
    words = s.split()

    for i in range(len(words)):
        if words[i] in exp_col_lst:  
            words[i] = target_color  
            break

    target_label[aug_label]['name'] = " ".join(words)
    aug_instr = target_label['action']
    if 'object' in target_label:
        aug_instr = f"{aug_instr} {target_label['object']['name']}"
    if 'target' in target_label:
        aug_instr = f"{aug_instr} to {target_label['target']['name']}"
    return aug_instr