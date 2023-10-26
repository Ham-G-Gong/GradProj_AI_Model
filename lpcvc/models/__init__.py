import copy
import torchvision.models as models
import torch.nn as nn

from lpcvc.models.fanet import FANet
from lpcvc.models.KD_SENet_stage3_fpn14_b111 import kd_senet_stage3_fpn14_b111
from lpcvc.models.stdc_teacher import FANet_se_stdc1

class BatchNorm2d(nn.BatchNorm2d):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, num_features, activation='none'):
        super(BatchNorm2d, self).__init__(num_features=num_features)
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'none':
            self.activation = lambda x:x
        else:
            raise Exception("Accepted activation: ['leaky_relu']")

    def forward(self, x):
        return self.activation(super(BatchNorm2d, self).forward(x))
    
# not KD || basic segmentation model training
def get_model(model_dict, nclass, loss_fn=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")
    param_dict["loss_fn"] = loss_fn
    param_dict['norm_layer'] = BatchNorm2d
    
    model = model(nclass=nclass, **param_dict)
    return model

# use KD || segmentation model training with KD || student model
def get_student_model(model_dict, nclass, 
                      sg_loss=None,
                      pi_loss=None,
                      pa_loss=None,
                      lambda_pa=0.5,
                      lambda_pi=10.0):
    
    name = model_dict["student"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("student")
    
    param_dict["sg_loss"] = sg_loss
    param_dict["pi_loss"] = pi_loss
    param_dict["pa_loss"] = pa_loss

    param_dict["lambda_pa"] = lambda_pa
    param_dict["lambda_pi"] = lambda_pi

    param_dict['norm_layer'] = BatchNorm2d
    
    model = model(nclass=nclass, **param_dict)
    return model

# use KD || segmentation model load with KD || teacher model
def get_teacher_model(model_dict, nclass):
    name = model_dict["teacher"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("teacher")
    param_dict['norm_layer'] = BatchNorm2d
    
    model = model(nclass=nclass, **param_dict)
    return model


def _get_model_instance(name):
        
    # -----------------------------------------------------------------
    # baseline model : fanet
    if name == "fanet":
        return {
            "fanet": FANet
        }[name]
    
    # -----------------------------------------------------------------
    # stdunet model : FANet + se module | backbone : resnet7
    if name == "kd_senet_stage3_fpn14_b111":
        return {
            "kd_senet_stage3_fpn14_b111": kd_senet_stage3_fpn14_b111
        }[name]
    
    # -----------------------------------------------------------------
    # teacher model : FANet + se module | backbone : stdc1
    if name == "fanet_se_stdc1":
        return {
            "fanet_se_stdc1": FANet_se_stdc1
        }[name]   
    
    else:
        raise ("Model {} not available".format(name))
