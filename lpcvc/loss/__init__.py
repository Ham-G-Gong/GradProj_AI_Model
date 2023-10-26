import logging
import torch
import functools

# segmentation model loss
from lpcvc.loss.loss import (
    CrossEntropyLoss,
    OhemCELoss2D,
)

# knowledge distillation loss
from lpcvc.loss.loss import  CriterionPixelWise, CriterionPairWiseforWholeFeatAfterPool

logger = logging.getLogger("lpcvc")

key2loss = {
    # segmentation loss
    "CrossEntropyLoss": CrossEntropyLoss,
    "OhemCELoss2D": OhemCELoss2D,

    #knowledge distillation loss
    "CriterionPixelWise" : CriterionPixelWise,
    "CriterionPairWiseforWholeFeatAfterPool" : CriterionPairWiseforWholeFeatAfterPool,
}


# not KD || basic segmentation model training
def get_loss_function(cfg):
    assert(cfg["loss"] is not None)
    loss_dict = cfg["loss"]
    loss_name = loss_dict["name"]
    loss_params = {k: v for k, v in loss_dict.items() if k != "name"}
    if loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(loss_name))

    if loss_name == "OhemCELoss2D":
        n_img_per_gpu = int(cfg["batch_size"]/torch.cuda.device_count())
        cropsize = cfg["train_augmentations"]["rcrop"]
        n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
        loss_params["n_min"] = n_min

    logger.info("Using {} with {} params".format(loss_name, loss_params))

    sg_loss = key2loss[loss_name](**loss_params)
    return sg_loss


# use KD || segmentation model training with KD
def get_dist_loss_function(cfg):

    assert(cfg["sg_loss"] is not None)
    assert(cfg["pi_loss"] is not None)
    assert(cfg["pa_loss"] is not None)


    sg_loss_dict = cfg["sg_loss"]
    pi_loss_dict = cfg["pi_loss"]
    pa_loss_dict = cfg["pa_loss"]


    sg_loss_name = sg_loss_dict["name"]
    pi_loss_name = pi_loss_dict["name"]
    pa_loss_name = pa_loss_dict["name"]


    sg_loss_params = {k: v for k, v in sg_loss_dict.items() if k != "name"}
    pi_loss_params = {k: v for k, v in pi_loss_dict.items() if k != "name"}
    pa_loss_params = {k: v for k, v in pa_loss_dict.items() if k != "name"}

    if sg_loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(sg_loss_name))
    if pi_loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(pi_loss_name))
    if pa_loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(pa_loss_name))

    logger.info("Using {} with {} params".format(sg_loss_name, sg_loss_params))
    logger.info("Using {} with {} params".format(pi_loss_name, pi_loss_params))
    logger.info("Using {} with {} params".format(pa_loss_name, pa_loss_params))


    if sg_loss_name == "OhemCELoss2D":
        n_img_per_gpu = int(cfg["batch_size"])
        cropsize = cfg["train_augmentations"]["rcrop"]
        n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
        sg_loss_params["n_min"] = n_min

    sg_loss = key2loss[sg_loss_name](**sg_loss_params)
    pi_loss = key2loss[pi_loss_name](**pi_loss_params)
    pa_loss = key2loss[pa_loss_name](**pa_loss_params)


    return sg_loss, pi_loss, pa_loss