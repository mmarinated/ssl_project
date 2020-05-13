"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""
import sys
sys.path.append("/scratch/mz2476/DL/project/fastai")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
# import ...
# from model_loader_utils import get_bounding_boxes_from_seg, to_np
from modelzoo import mmd_vae, vae
from ssl_project.vehicle_layout_prediction.bb_utils import ProcessSegmentationMaps

from ssl_project.perspective_transform.projecting import ProjectionToBEV
from ssl_project.paths import PATH_TO_REPO

# Put your transform function here, we will use it for our dataloader
def get_transform():
    return torchvision.transforms.ToTensor()

get_transform_task1, get_transform_task2 = torchvision.transforms.ToTensor(), torchvision.transforms.ToTensor()

def normalize_image(image_b3ww, stats=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):
    assert image_b3ww.ndim >= 3 and image_b3ww.shape[-3] == 3
    return (image_b3ww - stats[0][:, None, None]) / stats[1][:, None, None]

class ModelLoader():
    # Fill the information for your team
    team_name = 'team_featureMa.P.S'
    round_number = 1
    team_member = ['sc6957','kae358','mz2476'] # Shreyas, Philip & Marina
    contact_email = 'sc6957@nyu.edu' #kae358@nyu.edu, mz2476@nyu.edu

    def __init__(self, model_file='./team_MAPS_second_sub_checkpoint.pth.tar', *,
                threshold=0.5, device="cpu"):
        # You should
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        #
        self.road_model = mmd_vae()
        model_weights = torch.load(model_file, map_location="cpu")
        self.road_model.load_state_dict(model_weights['road_model_state_dict'])

        self.object_model_unet = torch.load(
            f"{PATH_TO_REPO}/pretrained_models/unet_v0.pth",
            map_location="cpu")
        
        self.road_model.to(device)
        self.object_model_unet.to(device)

        self.road_model.eval()
        self.object_model_unet.eval()
        
        self.process_segm = ProcessSegmentationMaps()
        
        self.proj = ProjectionToBEV()
        self.upsample_photo = nn.Upsample(size=(800, 800))
        self.threshold = threshold

    def get_bounding_boxes(self, samples_b63hw):
        """return tuple(bbs_n24)"""
        
        final_bbs = []
        for photos_63hw in samples_b63hw:
            segm_3WW = self.proj.get_warped_3WW(photos_63hw.cpu()).to(photos_63hw.device)
            segm_3ww = F.interpolate(segm_3WW[None], size=(200, 200))[0]
            segm_3ww = self._normalize_image(segm_3ww)
            fcast_2WW = self.upsample_photo(self.object_model_unet(segm_3ww[None]))[0]
            fcast_WW  = F.softmax(fcast_2WW, dim=-3)[1, :, :]
            bbs_k24 = self._segm_maps_to_bbs(fcast_WW, self.threshold)
            final_bbs.append(bbs_k24)
            
        return tuple(final_bbs)
            
            
#         for i in range(out.size()[0]):
#             bbs_k24 = self.process_segm.transform(out[i], threshold=0.4)
#             if len(bbs_k24) > 0:
#                 bbs_k24 = self.process_segm.convert_to_bb_space(bbs_k24, axis=-2)
#             else:
#                 bbs_k24 = torch.zeros((1,2,4))
#             bbs.append(bbs_k24)
#         return tuple(bbs)

    def _segm_maps_to_bbs(self, fcast_WW, threshold=0.5):
        bbs_k24 = self.process_segm.transform(fcast_WW, threshold=threshold)
        if len(bbs_k24) > 0:
            bbs_k24 = self.process_segm.convert_to_bb_space(bbs_k24, axis=-2)
        else:
            bbs_k24 = torch.zeros((1,2,4))
        return bbs_k24
    
    @staticmethod
    def _normalize_image(image_b3ww, 
                    mean_3=torch.Tensor([0.485, 0.456, 0.406]), 
                    std_3=torch.Tensor([0.229, 0.224, 0.225])):
        assert image_b3ww.ndim >= 3 and image_b3ww.shape[-3] == 3
        mean_3 = mean_3.to(device=image_b3ww.device)
        std_3  = std_3.to(device=image_b3ww.device)
        return (image_b3ww - mean_3[:, None, None]) / std_3[:, None, None]
    
    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        return self.road_model(samples)[1] > 0.5
        #return torch.rand(1, 800, 800)
