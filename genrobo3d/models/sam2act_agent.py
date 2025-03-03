from sam2act.models.modified_agent import SAM2Act_Agent2 as SAM2Act_Agent
from genrobo3d.utils.rotation_transform import RotationMatrixTransform
import torch
class SAM2Act_Agent2(SAM2Act_Agent):
    def __init__(self, *args, **kwargs):
        super(SAM2Act_Agent2, self).__init__(*args, **kwargs)
        self.rot_transform = RotationMatrixTransform()

    def _tran_loss(self,loss_log):
        losses = {
            'pos':loss_log['trans_loss'], 
            'rot':loss_log['rot_loss_x']+loss_log['rot_loss_y']+loss_log['rot_loss_z'], 
            'open':loss_log['grip_loss'], 
            'stop':loss_log['collision_loss'],
            'total':loss_log['total_loss']
        }
        return losses
    
    def prepare_batch(self, batch):
        device = self._device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch
    
    def __call__(self, batch, compute_loss=False, for_eval = False, **kwargs):
        '''
        final_pred_actions:(bs,max_traj_len,9)3 wpt, 4 rot, 1 open, 1 stop
        losses:['pos', 'rot', 'open', 'stop', 'total'],loss
        if compute_loss:
            return final_pred_actions, losses
        else:
            final_pred_actions
        '''
        gt_traj = batch['gt_trajs'].clone()
        rot = self.rot_transform.euler_to_quaternion(gt_traj[:,0,3:-1].reshape(-1,3).data.cpu()).float()
        wpt = gt_traj[:,0,:3].reshape(-1,3)
        grip = gt_traj[:,0,-1].reshape(-1,1)
        batch['gt_trajs'] = torch.cat([wpt,rot,grip],1)
        print(batch['gt_trajs'].shape)
        batch = self.prepare_batch(batch)
        if compute_loss:
            if not for_eval:
                final_pred_actions,losses = super(SAM2Act_Agent2, self).__call__(batch, 
                backprop=True)
                losses = self._tran_loss(losses)
                return final_pred_actions, losses
            else:
                final_pred_actions,losses = super(SAM2Act_Agent2, self)._eval(batch)
                losses = self._tran_loss(losses)
                return final_pred_actions, losses
        else:
            #只获取动作
            final_pred_actions,_ = super(SAM2Act_Agent2, self)._eval(batch)
            return final_pred_actions
        

    
    
if __name__ == '__main__':

    fake_batch = {
        'pc_fts': torch.rand(100, 6),
        'npoints_in_batch': [30, 70],
        'offset': torch.LongTensor([30, 100]),
        'txt_embeds': torch.rand(2, 512),
        'txt_lens': [1, 1],
        'ee_poses': torch.rand(2, 8),
        'gt_trajs': torch.rand(2, 5, 3+3+1),
    }
    model=SAM2Act_Agent2()
    outs = model(fake_batch, compute_loss=True)
    print(outs[1])