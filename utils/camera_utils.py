import torch
from torch import nn
import gc
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.slam_utils import image_gradient, image_gradient_mask


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        self.T_W = torch.eye(4, device=device)
        self.R = self.T_W[:3, :3]
        self.T = self.T_W[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)
    
    def reset_view_param(self):
        self.cam_rot_delta = nn.Parameter(
                torch.zeros(3, requires_grad=True, device=self.device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=self.device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=self.device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=self.device)
        )
    
    
    
    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)
        # self.T_W[:3, :3] = self.R 
        # self.T_W[:3, 3] = self.T   

    def update_RT_cpu(self, R, t):
        self.R = R.clone().to("cpu")
        self.T = t.clone().to("cpu")
        
    def update_RT_gpu(self, R, t):
        self.R = R.clone().cuda()
        self.T = t.clone().cuda()
        
    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if config["Dataset"]["type"] == "replica":
            row, col = 32, 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            for r in range(row):
                for c in range(col):
                    block = img_grad_intensity[
                        :,
                        r * int(h / row) : (r + 1) * int(h / row),
                        c * int(w / col) : (c + 1) * int(w / col),
                    ]
                    th_median = block.median()
                    block[block > (th_median * multiplier)] = 1
                    block[block <= (th_median * multiplier)] = 0
            self.grad_mask = img_grad_intensity
        else:
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None
    
    # def to_gpu(self) :
        
    #     self._xyz = nn.Parameter(
    #         torch.tensor(self._xyz, dtype=torch.float, device="cuda").requires_grad_(True)
    #     )
    #     self._features_dc = nn.Parameter(
    #         torch.tensor(self._features_dc, dtype=torch.float, device="cuda")
            
    #         .contiguous()
    #         .requires_grad_(True)
    #     )
    #     self._features_rest = nn.Parameter(
    #         torch.tensor(self._features_rest, dtype=torch.float, device="cuda")      
    #         .contiguous()
    #         .requires_grad_(True)
    #     )
    #     self._opacity = nn.Parameter(
    #         torch.tensor(self._opacity, dtype=torch.float, device="cuda").requires_grad_(
    #             True
    #         )
    #     )
    #     self._scaling = nn.Parameter(
    #         torch.tensor(self._scaling, dtype=torch.float, device="cuda").requires_grad_(True)
    #     )
    #     self._rotation = nn.Parameter(
    #         torch.tensor(self._rotation, dtype=torch.float, device="cuda").requires_grad_(True)
    #     ) 
    def viewpoints_to_gpu(self):
        self.cam_rot_delta = self.cam_rot_delta.requires_grad_(True)
        self.cam_trans_delta = self.cam_trans_delta.requires_grad_(True)
        self.exposure_a = self.exposure_a.requires_grad_(True)
        self.exposure_b = self.exposure_b.requires_grad_(True)  

    def viewpoints_to_cpu(self):
        tmp_original_image = self.original_image.detach().to("cpu")
        tmp_cam_rot_delta = self.cam_rot_delta.detach().to("cpu")
        tmp_cam_trans_delta = self.cam_trans_delta.detach().to("cpu")
        tmp_exposure_a = self.exposure_a.detach().to("cpu")
        tmp_exposure_b = self.exposure_b.detach().to("cpu")
        tmp_R = self.R.detach().to("cpu")
        tmp_T = self.T.detach().to("cpu")
        tmp_grad_mask = self.grad_mask.detach().to("cpu")
        del self.original_image
        del self.cam_rot_delta
        del self.cam_trans_delta
        del self.exposure_a
        del self.exposure_b
        del self.R
        del self.T
        del self.grad_mask
        torch.cuda.empty_cache()
        gc.collect()
        self.original_image = tmp_original_image
        self.cam_rot_delta = tmp_cam_rot_delta
        self.cam_trans_delta = tmp_cam_trans_delta
        self.exposure_a = tmp_exposure_a
        self.exposure_b = tmp_exposure_b
        self.R = tmp_R
        self.T = tmp_T
        self.grad_mask= tmp_grad_mask
    
    def img_to_cpu(self):
        tmp_original_image = self.original_image.detach().to("cpu")
        # tmp_cam_rot_delta = self.cam_rot_delta.detach().to("cpu")
        # tmp_cam_trans_delta = self.cam_trans_delta.detach().to("cpu")
        # tmp_exposure_a = self.exposure_a.detach().to("cpu")
        # tmp_exposure_b = self.exposure_b.detach().to("cpu")
        # tmp_R = self.R.detach().to("cpu")
        # tmp_T = self.T.detach().to("cpu")
        # tmp_grad_mask = self.grad_mask.detach().to("cpu")
        del self.original_image       
        torch.cuda.empty_cache()
        gc.collect()
        self.original_image = tmp_original_image      
        
    # def clean2(self):
    #     self.original_image.to("cpu")
    #     self.original_image = None       
    #     self.cam_rot_delta.detach().to("cpu")   
    #     self.cam_trans_delta.detach().to("cpu")
    #     self.exposure_a.detach().to("cpu")
    #     self.exposure_b.detach().to("cpu")
 