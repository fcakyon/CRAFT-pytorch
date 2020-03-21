import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np

import craft_text_detector.craft_utils as craft_utils
import craft_text_detector.imgproc as imgproc
import craft_text_detector.file_utils as file_utils
from craft_text_detector.models.craftnet import CRAFT

from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def load_craftnet_model(cuda: bool = False):
    # get craft net path
    home_path = str(Path.home())
    weight_path = os.path.join(home_path,
                               "craft_text_detector",
                               "weights",
                               "craft_mlt_25k.pth")
    # load craft net
    craft_net = CRAFT()  # initialize

    # check if weights are already downloaded, if not download
    weight_dir = os.path.dirname(weight_path)
    url="https://github.com/fcakyon/craft_text_detector/releases/download/v0.0.1/craft_mlt_25k.pth"
    if os.path.isfile(weight_path) is not True:
        print("Craft text detector weight will be downloaded to {}"
              .format(weight_path))

        file_utils.download(url=url, save_dir=weight_dir)

    # arange device
    if cuda:
        craft_net.load_state_dict(copyStateDict(torch.load(weight_path)))

        craft_net = craft_net.cuda()
        craft_net = torch.nn.DataParallel(craft_net)
        cudnn.benchmark = False
    else:
        craft_net.load_state_dict(copyStateDict(torch.load(weight_path,
                                                           map_location='cpu')))
    craft_net.eval()
    return craft_net


def load_refinenet_model(cuda: bool = False):
    # get refine net path
    home_path = str(Path.home())
    weight_path = os.path.join(home_path,
                               "craft_text_detector",
                               "weights",
                               "craft_refiner_CTW1500.pth")
    # load refine net
    from craft_text_detector.models.refinenet import RefineNet
    refine_net = RefineNet()  # initialize

    # check if weights are already downloaded, if not download
    weight_dir = os.path.dirname(weight_path)
    url="https://github.com/fcakyon/craft_text_detector/releases/download/v0.0.1/craft_refiner_CTW1500.pth"
    if os.path.isfile(weight_path) is not True:
        print("Craft text refiner weight will be downloaded to {}"
              .format(weight_path))

        file_utils.download(url=url, save_dir=weight_dir)

    # arange device
    if cuda:
        refine_net.load_state_dict(copyStateDict(torch.load(weight_path)))

        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
        cudnn.benchmark = False
    else:
        refine_net.load_state_dict(copyStateDict(torch.load(weight_path,
                                                            map_location='cpu')))
    refine_net.eval()
    return refine_net


def get_prediction(image,
                   craft_net,
                   refine_net=None,
                   text_threshold: float = 0.7,
                   link_threshold: float = 0.4,
                   low_text: float = 0.4,
                   cuda: bool = False,
                   canvas_size: int = 1280,
                   mag_ratio: float = 1.5,
                   poly: bool = True,
                   show_time: bool = False):
    """
    Arguments:
        image_path: path to the image to be processed
        output_dir: path to the results to be exported
        text_threshold: text confidence threshold
        link_threshold: link confidence threshold
        low_text: text low-bound score
        cuda: Use cuda for inference
        canvas_size: image size for inference
        mag_ratio: image magnification ratio
        poly: enable polygon type
        show_time: show processing time
        refiner: enable link refiner
        export_extra: export heatmap, detection points, box visualization
    """
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
            image, canvas_size, interpolation=cv2.INTER_LINEAR,
            mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = craft_net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text,
            poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    heatmap = imgproc.cvt2HeatmapImg(render_img)

    if show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, heatmap


#def detect_text(image_path: str,
#                output_dir: str = "output/",
#                text_threshold: float = 0.7,
#                link_threshold: float = 0.4,
#                low_text: float = 0.4,
#                cuda: bool = False,
#                canvas_size: int = 1280,
#                mag_ratio: float = 1.5,
#                poly: bool = False,
#                show_time: bool = False,
#                refiner: bool = False,
#                export_extra: bool = True):
#    """
#    Arguments:
#        image_path: path to the image to be processed
#        output_dir: path to the results to be exported
#        text_threshold: text confidence threshold
#        low_text: text low-bound score
#        link_threshold: link confidence threshold
#        cuda: Use cuda for inference
#        canvas_size: image size for inference
#        mag_ratio: image magnification ratio
#        poly: enable polygon type
#        show_time: show processing time
#        refiner: enable link refiner
#        export_extra: export heatmap, detection points, box visualization
#    """
#    # get models
#    craft_net = load_craftnet_model(cuda)
#    if refiner:
#        refine_net = load_refinenet_model(cuda)
#    else:
#        refine_net = None
#
#    t = time.time()
#
#    # load image
#    image = imgproc.read_image(image_path)
#
#    # perform text detection
#    bboxes, polys, heatmap = get_prediction(image,
#                                            craft_net,
#                                            refine_net,
#                                            text_threshold,
#                                            link_threshold,
#                                            low_text,
#                                            cuda,
#                                            canvas_size,
#                                            mag_ratio,
#                                            poly,
#                                            show_time)
#
#    # export detected text regions
#    file_utils.export_detected_regions(image_path, image, polys, output_dir)
#
#    if export_extra:
#        file_utils.export_extra_results(image_path,
#                                        image,
#                                        polys,
#                                        heatmap,
#                                        output_dir=output_dir)
#
#    print("elapsed time : {}s".format(time.time() - t))
