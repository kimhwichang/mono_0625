import time
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.submap_utils import Submap


class Refine(mp.Process):
    def __init__(self):
        super().__init__()
        self.requested_refine = True
        self.last_active_submap = None
        self.slam_queue = None
        self.refined_queue = None
       
    def color_refinement(self,submap_):
        Log("Starting color refinement")

        iteration_total = 5000 #26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            
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
    def request_refine(self, submap):
        msg = ["refine", self.last_active_submap]
        self.slam_queue.put(msg)
        self.requested_refine = True
        print("n->s  : tag = refined_submap")
        



    def run(self):
        while True:
            
            if self.refined_queue.empty():
               
                #request_init 일경우 backend의 init -> continue
                if self.requested_refine:
                    time.sleep(0.01)
                    continue
                self.color_refinement(self.last_active_submap)   
                self.request_refine(self.last_active_submap)       
               
                
            else:
                data = self.refined_queue.get()
                if data[0] == "refine":
                    print("f->r : tag = refine")
                    self.last_active_submap = data[1]
                    self.requested_refine = False
                    

               