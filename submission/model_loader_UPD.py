"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
# import ...
# from model_loader_utils import get_bounding_boxes_from_seg, to_np
from modelzoo import mmd_vae, vae
from ssl_project.vehicle_layout_prediction.bb_utils import ProcessSegmentationMaps


# Put your transform function here, we will use it for our dataloader
def get_transform():
    return torchvision.transforms.ToTensor()

get_transform_task1, get_transform_task2 = torchvision.transforms.ToTensor(), torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'team_featureMa.P.S'
    round_number = 1
    team_member = ['sc6957','kae358','mz2476'] # Shreyas, Philip & Marina
    contact_email = 'sc6957@nyu.edu' #kae358@nyu.edu, mz2476@nyu.edu

    def __init__(self, model_file='./team_MAPS_second_sub_checkpoint.pth.tar'):
        # You should
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        #
        self.road_model = mmd_vae()
        self.object_model = vae()

        model_weights = torch.load(model_file, map_location="cpu")

        self.road_model.load_state_dict(model_weights['road_model_state_dict'])
        self.object_model.load_state_dict(model_weights['object_model_state_dict'])

        self.road_model.cuda()
        self.object_model.cuda()

        self.road_model.eval()
        self.object_model.eval()
        
        self.process_segm = ProcessSegmentationMaps()

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        out,mu,var = self.object_model(samples, is_training=False)

        bbs = []
        for i in range(out.size()[0]):
            bbs_k24 = self.process_segm.transform(out[i], threshold=0.4)
            if len(bbs_k24) > 0:
                bbs_k24 = self.process_segm.convert_to_bb_space(bbs_k24, axis=-2)
            else:
                bbs_k24 = torch.zeros((1,2,4))
            bbs.append(bbs_k24)
        return tuple(bbs)

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        return self.road_model(samples)[1] > 0.5
        #return torch.rand(1, 800, 800)
