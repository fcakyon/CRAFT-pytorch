import os
import time

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np

import craft_utils as craft_utils
import imgproc as imgproc
import file_utils as file_utils
from craftnet import CRAFT

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


def load_weights(net, model_path: str, cuda: bool = False):
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(model_path)))

        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    else:
        net.load_state_dict(copyStateDict(torch.load(model_path,
                                                     map_location='cpu')))
    net.eval()
    return net


def get_models(cuda: bool = False, refiner: bool = False):
    # load craft net
    model_path = os.path.join("weights", "craft_mlt_25k.pth")
    # load craft net
    craft_net = CRAFT()  # initialize
    # arange device
    craft_net = load_weights(craft_net, model_path, cuda)

    # load refine net
    if refiner:
        model_path = os.path.join("weights", "craft_refiner_CTW1500.pth")
        # load net
        from refinenet import RefineNet
        refine_net = RefineNet()
        # arange device
        refine_net = load_weights(refine_net, model_path, cuda)
        poly = True
    else:
        refine_net = None
        poly = False

    return craft_net, refine_net, poly


def get_prediction(craft_net,
                   refine_net,
                   image,
                   text_threshold: float = 0.7,
                   link_threshold: float = 0.4,
                   low_text: float = 0.4,
                   cuda: bool = False,
                   canvas_size: int = 1280,
                   mag_ratio: float = 1.5,
                   poly: bool = False,
                   show_time: bool = False):
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
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def detect_text(image_path: str,
                output_dir: str = "output/",
                text_threshold: float = 0.7,
                link_threshold: float = 0.4,
                low_text: float = 0.4,
                cuda: bool = False,
                canvas_size: int = 1280,
                mag_ratio: float = 1.5,
                poly: bool = False,
                show_time: bool = False,
                refiner: bool = False,
                export_extra: bool = True):
    """
    Arguments:
        image_path: path to the image to be processed
        output_dir: path to the results to be exported
        text_threshold: text confidence threshold
        low_text: text low-bound score
        link_threshold: link confidence threshold
        cuda: Use cuda for inference
        canvas_size: image size for inference
        mag_ratio: image magnification ratio
        poly: enable polygon type
        show_time: show processing time
        refiner: enable link refiner
        export_extra: export score map, detection points, box visualization
    """

    # create output dir
    file_utils.create_dir(output_dir)

    # get models
    craft_net, refine_net, poly = get_models(cuda, refiner)

    t = time.time()

    # load image
    image = imgproc.loadImage(image_path)

    # perform text detection
    bboxes, polys, score_text = get_prediction(craft_net,
                                               refine_net,
                                               image,
                                               text_threshold,
                                               link_threshold,
                                               low_text,
                                               cuda,
                                               canvas_size,
                                               mag_ratio,
                                               poly,
                                               show_time)

    # export detected text regions
    file_utils.export_detected_regions(image_path, image, polys, output_dir)

    if export_extra:
        # export score map
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = os.path.join(output_dir, "res_" + filename + '_mask.jpg')
        cv2.imwrite(mask_file, score_text)

        # export detected points and box visualization
        file_utils.export_extra_results(image_path,
                                        image[:, :, ::-1],
                                        polys,
                                        output_dir=output_dir)

    print("elapsed time : {}s".format(time.time() - t))
