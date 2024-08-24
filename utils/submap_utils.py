import torch
import gc
from torch import nn
from munch import munchify
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.slam_utils import image_gradient, image_gradient_mask
from gaussian_splatting.scene.gaussian_model import GaussianModel

class Submap(nn.Module):
    def __init__(self,config,device,id,first_):
        
        super().__init__()
        self.uid = id
        self.first_ = first_ 
        self.config = config
        self.device = device
        self.model_params = munchify(config["model_params"])
        self.opt_params = munchify(config["opt_params"])
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        #self.pipeline_params = munchify(config["pipeline_params"])
        self.model_params.sh_degree = 3 if self.use_spherical_harmonics else 0
        self.gaussians = GaussianModel(self.model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        self.cameras_extent = 6.0
        self.anchor_frame = None      
        self.T_CW = torch.eye(4)
        self.T_WC = torch.eye(4)
        self.last_pose = torch.eye(4)
    
        self.initialilzed = False
        self.viewpoints = {}
        self.kf_idx =[]
        self.updated_idx =[]
        self.pose_list =[]        
        self.current_window=[]
        self.occ_aware_visibility={}
        self.keyframe_optimizers = None        
        self.set_hyperparams()
        # self.device = "cuda"
        # self.dtype = torch.float32
    
    def submap_to_cpu(self, submap):
        for keys,views in submap.viewpoints.items():
            views.viewpoints_to_cpu()
            self.occ_aware_visibility[keys] = None
            
            torch.cuda.empty_cache()
        self.gaussians.reset()             

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
       
    
    def initialize_(self, cur_view, last_anchor_pose = None,last_kf = None):
        if (self.first_):
            print("cur_idx = %i"%cur_view.uid)       
            self.kf_idx.append(cur_view.uid)
            # self.set_anchor_frame(cur_view)
            self.set_anchor_frame_pose(cur_view)
            self.set_anchor_frame_pose_inv()         
            self.viewpoints[cur_view.uid] = cur_view
            self.current_window.append(cur_view.uid)    
            self.pose_list.append(self.T_CW)
        else :
            
            print("cur_idx___ = %i"%cur_view.uid)           
            self.kf_idx.append(cur_view.uid)            
            # self.set_anchor_frame(cur_view)
            if last_anchor_pose is None :
                self.set_anchor_frame_pose(cur_view)
                self.set_anchor_frame_pose_inv()
            else :
                self.set_anchor_frame_pose(cur_view, last_anchor_pose)
                self.set_anchor_frame_pose_inv()

            # print(cur_view.R)
            # print(cur_view.T)
            
            temp_T = torch.eye(4)
            temp_R= temp_T[:3,:3]
            temp_t = temp_T[:3,3]
            cur_view.update_RT(temp_R,temp_t)
            
            # print(cur_view.R)           
            # print(cur_view.T)           

            self.viewpoints[cur_view.uid] = cur_view     
            self.current_window.append(cur_view.uid)            
            self.pose_list.append(self.T_CW)
            # self.pose_list.append(cur_view.T_W)     
    def gen_pose_matrix(self,R, T):
        pose = torch.eye(4)
        pose[0:3, 0:3] = R.to("cpu")
        pose[0:3, 3] = T.to("cpu")
        # print(pose)
        return pose
       
        
    def local_BA(self) :
        return None
    
    def set_anchor_frame(self,viewpoint) :
        self.anchor_frmae = viewpoint
    
    
    def set_anchor_frame_pose(self,viewpoint,last_anchor_pose=None) :
        
        self.T_CW[:3, :3] = viewpoint.R.clone() 
        self.T_CW[:3, 3] = viewpoint.T.clone()  
        self.original_TCW = self.T_CW.clone()
        if last_anchor_pose is not None:   
            self.T_CW = self.T_CW@last_anchor_pose      
            self.original_TCW = self.T_CW .clone()   
    
        
    def set_anchor_frame_pose_inv(self) :
        
        self.T_WC = self.T_CW.clone().inverse()
            
    def get_kf_idx(self):
        return self.kf_idx
    
    def get_anchor_frame(self) :
        return self.anchor_frame
    
    def get_anchor_frame_idx(self) :
        return self.anchor_frame
    
    def get_anchor_frame_pose(self) :
        return self.T_CW
    
    def get_anchor_frame_pose_inverse(self) :
        return self.T_WC
    
    def get_last_frame(self):
        # size_ = len(self.kf_idx)
        return self.viewpoints[self.kf_idx[-1]]
    
    def get_last_frame_idx(self):
        
        return self.kf_idx[-1]
    
    def get_win_size(self):
        return len(self.current_window)
    
    def get_submap_size(self):
        return len(self.kf_idx)
    

    def merge(self,submap_1 , submap_2) :
        #to do
        return True

    # if (self.lastkf):
    #     print("last kf id = %i" %last_kf.uid)
    #     self.kf_idx.append(last_kf.uid)
    #     self.kf_idx.append(cur_view.uid)
        
    #     if last_anchor_pose is None :
    #         self.set_anchor_frame_pose(last_kf)
    #         self.set_anchor_frame_pose_inv()
    #     else :
    #         self.set_anchor_frame_pose(last_kf, last_anchor_pose)
    #         self.set_anchor_frame_pose_inv()                
                    
    #     last_inv_T = torch.eye(4)
    #     last_inv_T[:3,:3] = last_kf.R.clone()
    #     last_inv_T[:3,3] = last_kf.T.clone()
    #     last_inv_T = last_inv_T.inverse()
    #     cur_T = torch.eye(4)
    #     cur_T[:3,:3] = cur_view.R.clone()
    #     cur_T[:3,3] = cur_view.T.clone()                
    #     cur_T_ = cur_T.clone()@last_inv_T.clone()
    #     cur_R = cur_T_[:3,:3]
    #     cur_t = cur_T_[3,:3]
    #     cur_view.update_RT(cur_R,cur_t)                
    #     temp_T= torch.eye(4)
    #     temp_R= temp_T[:3,:3]
    #     temp_t = temp_T[:3,3]                 
    #     last_kf.update_RT(temp_R,temp_t)        
        
    #     self.viewpoints[last_kf.uid] = last_kf
    #     self.viewpoints[cur_view.uid] = cur_view     
    #     self.current_window.append(last_kf.uid)
    #     self.current_window = [cur_view.uid]+self.current_window         
    #     self.pose_list.append(self.T_CW)
    