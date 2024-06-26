import torch
from torch import nn
from munch import munchify
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.slam_utils import image_gradient, image_gradient_mask
from gaussian_splatting.scene.gaussian_model import GaussianModel

class Submap(nn.Module):
    def __init__(self,config,id,first_):
        
        super().__init__()
        self.idx = 1 #현재 몇개 째 들어왔는지
        self.uid = id
        self.first_ = first_
        self.config = config
        
        self.model_params = munchify(config["model_params"])
        self.opt_params = munchify(config["opt_params"])
        #self.pipeline_params = munchify(config["pipeline_params"])

        self.gaussians = GaussianModel(self.model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        self.cameras_extent = 6.0
        self.anchor_frame = None      
        self.T_CW = torch.eye(4)
        self.initialilzed = False
        self.viewpoints = {}
        self.kf_idx =[]
        self.pose_list =[]
        self.current_window=[]
        self.occ_aware_visibility={}
        self.keyframe_optimizers = None        
        self.set_hyperparams()
        # self.device = "cuda"
        # self.dtype = torch.float32
        
        

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
        
       
    def initialize_(self, cur_view, prev_view =None):
        if (self.first_):
            self.kf_idx.append(cur_view.uid)
            self.viewpoints[cur_view.uid] = cur_view
            self.current_window.append(cur_view.uid)            
            self.set_anchor_frame(cur_view)
            self.set_anchor_frame_pose(cur_view)
            self.pose_list.append(self.T_CW)
        else :
            self.kf_idx.append(prev_view.uid)
            self.kf_idx.append(cur_view.uid)
            self.viewpoints[prev_view.uid] = prev_view
            self.viewpoints[cur_view.uid] = cur_view
            self.current_window.append(prev_view.uid)
            self.current_window.append(cur_view.uid)
            self.set_anchor_frame(prev_view)
            self.set_anchor_frame_pose(prev_view)
            self.pose_list.append(self.T_CW)
            self.pose_list.append(cur_view.T_W)
            self.idx+=1        
        
    def local_BA(self) :
        return None
    
    def set_anchor_frame(self,viewpoint) :
        self.anchor_frmae = viewpoint
    
    def set_anchor_frame_pose(self,viewpoint) :
        
        self.T_CW = viewpoint.T_W
            
    def get_kf_idx(self):
        return self.kf_idx
    
    def get_anchor_frame(self) :
        return self.anchor_frame
    
    def get_anchor_frame_pose(self) :
        return self.T_CW
    
    
    def get_last_frame(self):
        size_ = len(self.kf_idx)
        return self.viewpoints[self.kf_idx[-1]]
    
    def get_win_size(self):
        return len(self.current_window)
    

    def merge(self,submap_1 , submap_2) :
        
        self.anchor_frame = None      
        self.T_CW = torch.eye(4)
        self.initialilzed = False
        self.viewpoints = {}
        self.kf_idx_list =[]
        self.occ_aware_visibility={}
        self.keyframe_optimizers = None   
        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()
