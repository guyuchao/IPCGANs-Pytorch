import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def tensor2pil(tensor,norm="zero_center"):
    assert len(tensor.size())==3 and tensor.size()[0]==3,"need to be [3,h,w] tensor"
    assert norm is "zero_center" or norm is "normalization" or norm is None,"only support zero_center or normalization"
    if norm is "zero_center":
        tensor=(tensor+1)*127.5
    np_tensor=tensor.detach().cpu().numpy().transpose(1,2,0)
    return Image.fromarray(np_tensor.astype(np.uint8))

class Img_to_zero_center(object):
    def __int__(self):
        pass
    def __call__(self, t_img):
        '''
        :param img:tensor be 0-1
        :return:
        '''
        t_img=(t_img-0.5)*2
        return t_img

class Reverse_zero_center(object):
    def __init__(self):
        pass
    def __call__(self,t_img):
        t_img=t_img/2+0.5
        return t_img

if __name__=="__main__":
    print(str(20+1*10))