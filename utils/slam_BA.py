import time
import random
import numpy as np
from munch import munchify
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from gaussian_splatting.gaussian_renderer import render
from utils.logging_utils import Log
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim , l2_loss , l1_loss_log_pow , l2_loss_log
from utils.pose_utils import update_pose,update_pose_
from utils.multiprocessing_utils import clone_obj
from utils.slam_utils import get_loss_mapping, get_loss_tracking , depth_reg
from utils.eval_utils import eval_ate, save_gaussians, eval_rendering_ ,eval_ate_
from utils.submap_utils import Submap


class BA(mp.Process):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.requested_refine = True
        self.last_active_submap = None
        self.before_ba_queue = None
        self.pipeline_params = None    
        self.opt_params = munchify(config["opt_params"])   
        self.submap_list =[]
        self.dataset = None
        self.cameras=dict()
        self.kf_indices=[]
        self.save_dir = self.config["Results"]["save_dir"]
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]
        self.flag = False
        
    def check_mem(self,tag):
       
        print( tag+ f" Reserved Memory : {torch.cuda.memory_reserved()/1024.0/1024.0:.2f}MB" )
        print( tag+ f" Allocated Memory : {torch.cuda.memory_allocated()/1024.0/1024.0:.2f}MB")

    
    def color_refinement(self,submap_,iter=0):
        Log("Starting color refinement")

        iteration_total = iter #26000
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
  
    def pose_refine(self ,refine_iter):        

        for viewpoint in self.last_active_submap.viewpoints.values():
            if viewpoint.uid ==self.last_active_submap.kf_idx[0]:
                continue
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
            
            for tracking_itr in range(refine_iter):  
                
                render_pkg = render(
                    viewpoint, self.last_active_submap.gaussians, self.pipeline_params, self.last_active_submap.background
                )
                image, depth, opacity = (
                    render_pkg["render"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )               
                pose_optimizer.zero_grad()
                # gt_image = viewpoint.original_image.cuda()
                # _, h, w = gt_image.shape
                # mask_shape = (1, h, w)
                # rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
                # rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
                # rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
                # loss_i = (opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)).mean()
                # # loss_i = (1.0 - self.opt_params.lambda_dssim) * Ll1 + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))          
            
                # log_ = True
                # loss_d = 0
                # # log_ = False        
                # gt_depth = torch.Tensor(viewpoint.depth).unsqueeze(0).cuda()
                # depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
                # opacity_mask = (opacity > 0.95).view(*depth.shape)  
                # depth_mask = depth_pixel_mask * opacity_mask
                # depth = depth*depth_mask
                # gt_depth = gt_depth*depth_mask   
                # if log_ :
                #     gt_depth[torch.where(gt_depth<=1)] =1
                #     depth[torch.where(depth<=1)] =1             
                #     d1 = l1_loss_log_pow(depth,gt_depth)        
                #     d2 = l2_loss_log(depth, gt_depth)               
                #     loss_d = torch.sqrt(d2-0.85*d1)     
                    
                #     # depth = torch.log(depth)
                #     # gt_depth = torch.log(gt_depth)          
                #     # dr = depth_reg(depth,gt_depth)           
                #     # ss = (1.0 - ssim(depth, gt_depth))      
                #     # loss_d =  0.6*d1 + 0.2* dr+ss           
                # else :
                #     d1 = l1_loss(depth,gt_depth) 
                #     loss_d =  0.6*d1 + 0.2* depth_reg(depth,gt_depth)+(1.0 - ssim(depth, gt_depth))
                # loss_tracking =loss_i*0.9+0.1*loss_d
                
                loss_tracking = get_loss_tracking(
                    self.config, image, depth, opacity, viewpoint
                )            
                loss_tracking.backward()

                with torch.no_grad():
                    pose_optimizer.step()
                    # converged = update_pose(viewpoint)               
                    converged = update_pose_(viewpoint) 
                if converged:       
                    print("[converge]trcaking_itr =  %i" %tracking_itr)   
                    break          
            
            return 0
    
    def submap_to_cpu(self, submap):
        # msg = ["reset_mem"]        
        # self.backend_queue.put(msg)
        # print("submap_to_cpu0")
        for keys,views in submap.viewpoints.items():
            self.cameras[keys].viewpoints_to_cpu()
            views.viewpoints_to_cpu()
        # print("submap_to_cpu1")
        for idx in submap.current_window :
            submap.occ_aware_visibility[idx] = None        
            torch.cuda.empty_cache()
        # print("submap_to_cpu2")
        # submap.gaussians.reset()
        # print("submap_to_cpu3")
 
    def eval_(self, tag_):
        eval_ate_(                        
            self.last_active_submap,
            tag_,
            self.save_dir,
            monocular=self.monocular,
        )  
        eval_rendering_(
            self.cameras,
            self.last_active_submap.gaussians,
            self.dataset,        
            self.pipeline_params,
            self.last_active_submap.background,
            kf_indices=self.last_active_submap.kf_idx,
            tag_=tag_,
        )
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
                print("not empty")
                data = self.before_ba_queue.get()
                if data[0] == "frame":
                    print("f->B : tag = frame")
                    viewpoint = data[1]
                    self.cameras[viewpoint.uid] = viewpoint      
                    print("[BA] frame num = %i" %len(self.cameras.keys()))
                    # for view  in self.cameras.values() :
                    #     print(" %i"%view.uid,end="")   
                    
                elif data[0] == "submap":
                    print("f->B : tag = submap [initialized]")
                    self.last_active_submap = data[1] 
                    init_idx = self.last_active_submap.kf_idx[0]
                    init_view = self.last_active_submap.viewpoints[init_idx]
                    self.cameras[init_idx] = init_view
                    self.kf_indices.append(init_idx)
                    self.flag = True
                elif data[0] == "keyframe":
                    print("f->B : tag = keyframe")
                    # self.check_mem("BA")
                    viewpoint = data[1]                   
                    self.last_active_submap.viewpoints[viewpoint.uid]= clone_obj(viewpoint)
                    self.last_active_submap.kf_idx.append(viewpoint.uid)
                    self.cameras[viewpoint.uid] = viewpoint      
                    self.kf_indices.append(viewpoint.uid)
                    print("[BA] frame num = %i" %len(self.cameras.keys()))
                    print("[BA] key frame idx = ",end="")
                    for idx  in self.kf_indices :
                        print(" %i"%idx,end="")
                    print(" [%i]"%len(self.kf_indices))                       
                    if ( len(self.last_active_submap.kf_idx) >=5 and len(self.kf_indices) % self.save_trj_kf_intv == 0):
                        Log("[BA] Evaluating ATE at frame: ", viewpoint.uid)
                        # print("len of kf indices = %i "%len(self.kf_indices))
                        # self.eval_("before")
                        eval_ate(                        
                            self.submap_list,
                            self.last_active_submap,
                            self.save_dir,
                            viewpoint.uid,
                            False,
                            monocular=self.monocular,
                            tag="BA"
                        )  
                        # rendering_result = eval_rendering(
                        #     self.cameras,                 
                        #     self.last_active_submap.gaussians,
                        #     self.dataset,
                        #     self.save_dir,
                        #     self.pipeline_params,
                        #     self.last_active_submap.background,
                        #     kf_indices = self.last_active_submap.kf_idx ,
                        #     iteration="before_opt",
                        # )
               
                elif data[0] == "final":
                    print("f->B : tag = submap [finished]")
                    gaussian_ = data[1]                                             
                    self.last_active_submap.gaussians = gaussian_    
                    print("[BA] active_submap size = %i" %self.last_active_submap.get_submap_size())
                    print("[BA] active sub map kf_idx_num = %i" %len(self.last_active_submap.kf_idx)) 
                    print("[BA] active sub map viewpoints num = %i" %len(self.last_active_submap.viewpoints.keys()))                   
                    # self.eval_("ba1_submap #%i"%self.last_active_submap.uid) 
                    self.last_active_submap.gaussians.training_setup_ba(self.opt_params)
                    self.eval_("before_color_refine_submap #%i"%self.last_active_submap.uid)      
                    self.color_refinement(self.last_active_submap,1000)  
                    self.eval_("after_color_refine_submap #%i"%self.last_active_submap.uid)  
                    self.pose_refine(100)   
                    self.eval_("after_pose_refine_submap #%i"%self.last_active_submap.uid)  
                    self.submap_to_cpu(self.last_active_submap) 
                    # self.push_to_frontend()
                    
                    self.submap_list.append(self.last_active_submap)
                    self.last_active_submap = None
                    self.flag= False
                    # self.requested_refine = False
                
                elif data[0]=="update":
                    print("f->B : tag = update")
                    if not self.flag:
                        continue
                    updated_frame = data[1]
                    for kf_id, kf_R, kf_T in updated_frame:                           
                        self.last_active_submap.viewpoints[kf_id].update_RT_gpu(kf_R.clone(), kf_T.clone()) 
                        self.cameras[kf_id].update_RT_gpu(kf_R.clone(), kf_T.clone()) 
                                
                elif data[0]=="end":
                    print("f->B : tag = all data finished")                   
                    gaussian_ = data[1]                      
                    self.last_active_submap.gaussians = gaussian_    
                    # self.eval_("be_submap #%i"%self.last_active_submap.uid)      
                    cr0 = time.time()
                    self.color_refinement(self.last_active_submap,1000)  
                    cr1 = time.time()
                    print("cr time = "+str(cr1-cr0))
                    # self.eval_("af_submap #%i"%self.last_active_submap.uid)                       
                    # self.pose_refine_t(self.last_active_submap)   
                    self.submap_to_cpu(self.last_active_submap) 
                    # self.push_to_frontend()                    
                    self.submap_list.append(self.last_active_submap)                  
                    break
                else :
                    Log("Wrong Queue tag!")  
                    continue
                    

               
               
               # print("[BA] active sub map kf_idx_num [kf_idx] = ",end="")
                    # for i  in self.last_active_submap.kf_idx :
                    #     print(" %i"%i,end="")
                    # print(" ")
                    # print("[BA] active sub map idx [viewpoints] = ",end="")
                    # for idx in self.last_active_submap.viewpoints.keys() :
                    #     print(" %i"%idx,end="")
                    # print(" ")