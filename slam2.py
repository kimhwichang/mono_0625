import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
import random
import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians_
from utils.logging_utils import Log
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend2 import BackEnd
from utils.slam_frontend2 import FrontEnd



class SLAM:
    def __init__(self, config, save_dir=None):

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)           
        start.record()
     
        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )    
        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0     
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )
      
        # bg_color = [0, 0, 0]
        # self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.submap_list = []
        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()
     
        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.use_gui = self.use_gui
        # self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.opt_params = self.opt_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        #self.frontend.refined_queue = refined_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        # self.backend.gaussians = self.gaussians
        # self.backend.background = self.background
        # self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode        
   
        # self.backend.set_hyperparams()        
        backend_process = mp.Process(target=self.backend.run)            
        backend_process.start()       
        self.frontend.run()    
        backend_queue.put(["pause"])
        
        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = 0.1
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")
        total_submap_num = len(self.frontend.submap_list)
        if total_submap_num ==0 :
            total_submap_num=1
        print("submap num = %i"%total_submap_num)
        print("final kf num = %i" %len(self.frontend.kf_indices))
        if self.eval_rendering:
            ATE = eval_ate(
                    self.frontend.submap_list,
                    self.frontend.active_submap,
                    self.save_dir,
                    0,
                    final=True,
                    monocular=self.monocular,
                )
            # total_submap_num = len(self.frontend.submap_list)
            total_psnr = 0
            total_ssim = 0
            total_lpips = 0
            total_frame_num = 0
            count =0
            for submap_ in self.frontend.submap_list:
                self.gaussians = submap_.gaussians
                kf_indices = submap_.kf_idx    
                # anchor_frame_matrix = submap_.get_anchor_frame_pose()
                rendering_result = eval_rendering(
                    self.frontend.cameras,
                    # anchor_frame_matrix,
                    self.gaussians,
                    self.dataset,
                    self.save_dir,
                    self.pipeline_params,
                    submap_.background,
                    kf_indices=kf_indices,
                    iteration="before_opt",
                )
                total_psnr+=rendering_result["mean_psnr"]*rendering_result["total frame num"]
                total_ssim +=rendering_result["mean_ssim"]*rendering_result["total frame num"]
                total_lpips +=rendering_result["mean_lpips"]*rendering_result["total frame num"]  
                total_frame_num +=rendering_result["total frame num"]
                save_gaussians_(submap_.gaussians, self.save_dir, count, final=False)
                count+=1
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                total_psnr/total_frame_num,
                total_ssim/total_frame_num,
                total_lpips/total_frame_num,
                ATE,
                FPS,
            )
            

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()
            # # backend_queue.put(["color_refinement"])
            # frontend_queue.put(["color_refinement"])
            # while True:
            #     if frontend_queue.empty():
            #         time.sleep(0.01)
            #         continue
            #     data = frontend_queue.get()
            #     if data[0] == "sync_backend" and frontend_queue.empty():
            #         # gaussians = data[1]
            #         # self.gaussians = gaussians
            #         break
            print("before_psnr = %f" %float(total_psnr/total_frame_num))
            # self.frontend.color_refinement()
            # total_psnr = 0
            # total_ssim = 0
            # total_lpips = 0
            # for submap_ in self.frontend.submap_list:
            #     self.gaussians = submap_.gaussians
            #     kf_indices = submap_.kf_idx    
            #     rendering_result = eval_rendering(
            #         self.frontend.cameras,
            #         self.gaussians,
            #         self.dataset,
            #         self.save_dir,
            #         self.pipeline_params,
            #         submap_.background,
            #         kf_indices=kf_indices,
            #         iteration="final",
            #     )
            #     total_psnr+=rendering_result["mean_psnr"]
            #     total_ssim +=rendering_result["mean_ssim"]
            #     total_lpips +=rendering_result["mean_lpips"]
            # columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            # metrics_table = wandb.Table(columns=columns)
            # metrics_table.add_data(
            #     "After",
            #     total_psnr/total_submap_num,
            #     total_ssim/total_submap_num,
            #     total_lpips/total_submap_num,
            #     ATE,
            #     FPS,
            # )
            # print("final_psnr = %f" %float(total_psnr/total_submap_num))
            # wandb.log({"Metrics": metrics_table})
           

        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass
   

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
