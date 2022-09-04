import time
import tkinter

import sys
sys.path.insert(1, "D:/Workspace/DATN/code")

from utils.transforms import test_transforms
from models.backbones.mobilenet_v2 import Mobilenet_v2
from utils.utils import embedding_extraction, get_embedding
from configs.config import DEVICE
from utils.transforms import test_transforms
from bson.binary import Binary
import pickle
import torch

def get_hand_info_from_image(image_path: str, name: str):
    start = time.time()
    hand_tensor1, fig1 = embedding_extraction(
        image_path=image_path,
        device=DEVICE,
        transforms=test_transforms,
        draw=True, 
        name=name
    )
    hand_tensor1 = torch.from_numpy(hand_tensor1[0])
    print("Lấy thông tin lòng bàn tay thành công trong thời gian {}s".format(round (time.time() - start)))
    hand_numpy = hand_tensor1.cpu().detach().numpy()
    return Binary(pickle.dumps(hand_numpy, protocol=2)), fig1



    
    
