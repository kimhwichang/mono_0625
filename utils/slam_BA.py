import time
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from gaussian_splatting.gaussian_renderer import render
from utils.logging_utils import Log
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim , l2_loss , l1_loss_log_pow , l2_loss_log
from utils.pose_utils import update_pose
from utils.multiprocessing_utils import clone_obj
from utils.slam_utils import get_loss_mapping, get_loss_tracking , depth_reg
from utils.submap_utils import Submap


class BA(mp.Process):
    def __init__(self):
        super().__init__()
        self.requested_refine = True
        self.last_active_submap = None
        self.before_ba_queue = None
        self.after_ba_queue = None
        self.pipeline_params = None       
        self.submap_list =[]
        self.cameras=dict()
        
    def color_refinement(self,submap_):
        Log("Starting color refinement")

        iteration_total = 500 #26000
        # for iteration in tqdm(range(1, iteration_total + 1)):
        for iteration in range(1, iteration_total + 1):
            
            viewpoint_idx_stack = list(submap_.viewpoints.keys())
            #print(viewpoint_idx_stack)
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            #print(viewpoint_cam_idx)
            viewpoint_cam = submap_.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, submap_.gaussians, self.pipeline_params, submap_.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - submap_.opt_params.lambda_dssim) * (
                Ll1
            ) + submap_.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                submap_.gaussians.max_radii2D[visibility_filter] = torch.max(
                    submap_.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                submap_.gaussians.optimizer.step()
                submap_.gaussians.optimizer.zero_grad(set_to_none=True)
                submap_.gaussians.update_learning_rate(iteration)
        
        Log("Map refinement done")    
   
        return False
  
    def pose_refine_t(self ,refine_iter,viewpoint,use_prev = False):
        
        if use_prev:
            prev_kf = self.active_submap.get_last_frame()
            viewpoint.update_RT(prev_kf.R.clone(),prev_kf.T.clone())
       
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )        
        pose_optimizer = torch.optim.Adam(opt_params)
        
        for tracking_itr in range(refine_iter):  #self.tracking_itr_num):
            
            render_pkg = render(
                viewpoint, self.active_submap.gaussians, self.pipeline_params, self.active_submap.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )               
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )            
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)               
    
            if converged:       
                # print("[converge]trcaking_itr =  %i" %tracking_itr)   
                break          
        
        return 0
    
    def submap_to_cpu(self, submap):
        # msg = ["reset_mem"]        
        # self.backend_queue.put(msg)
        print("submap_to_cpu0")
        for views in submap.viewpoints.values():
            # self.cameras[keys].viewpoints_to_cpu()
            views.viewpoints_to_cpu()
        print("submap_to_cpu1")
        for idx in submap.current_window :
            submap.occ_aware_visibility[idx] = None        
            torch.cuda.empty_cache()
        print("submap_to_cpu2")
        submap.gaussians.reset()
        print("submap_to_cpu3")   
        
    def push_to_frontend(self):
        msg = ["after_ba", clone_obj(self.last_active_submap)]
        self.after_ba_queue.put(msg)
        print("r->f  : tag = after ba submap to frontend")
    
    def run(self):
        while True:
            
            if self.before_ba_queue.empty():
                # print("empty")
                #request_init 일경우 backend의 init -> continue
                if self.requested_refine:
                    time.sleep(0.01)
                    continue   
                         
                # continue  
            else:
                # print("not empty")
                data = self.before_ba_queue.get()
                if data[0] == "before_ba":
                    print("f->r : tag = before ba")
                    self.last_active_submap = data[1]
                    self.color_refinement(self.last_active_submap)   
                    # self.pose_refine_t(self.last_active_submap)   
                    self.submap_to_cpu(self.last_active_submap) 
                    # self.push_to_frontend()
                    self.submap_list.append(self.last_active_submap)
                    # self.requested_refine = False
                else :
                    Log("Wrond Queue tag!")  
                    continue
                    

               