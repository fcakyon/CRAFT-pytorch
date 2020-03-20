# -*- coding: utf-8 -*-
import os
import cv2
import copy
import numpy as np


def create_dir(_dir):
    """
    Creates given directory if it is not present.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls


def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if (ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or
                    ext == '.png' or ext == '.pgm'):
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files


def export_detected_region(image, points, file_path):
    # points should have 1*4*2  shape
    if len(points.shape) == 2:
        points = np.array([np.array(points).astype(np.int32)])

    # create mask with shape of image
    mask = np.zeros(image.shape[0:2], dtype=np.uint8)

    # method 1 smooth region
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

    # method 2 not so smooth region
    # cv2.fillPoly(mask, points, (255))

    res = cv2.bitwise_and(image, image, mask=mask)
    rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    # export corpped region
    cv2.imwrite(file_path, cropped)


def export_detected_regions(image_path, image, polys,
                            output_dir: str = "output/"):
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)

    # get file name
    file_name, file_ext = os.path.splitext(os.path.basename(image_path))

    # create crops dir
    crops_dir = os.path.join(output_dir, file_name + "_crops")
    create_dir(crops_dir)

    for ind, poly in enumerate(polys):
        file_path = os.path.join(crops_dir, "crop_" + str(ind) + ".png")
        export_detected_region(image, points=poly, file_path=file_path)


def export_extra_results(image_path,
                         image,
                         boxes,
                         output_dir='output/',
                         verticals=None,
                         texts=None):
    """ save text detection result one by one
    Args:
        image_path (str): image file name
        image (array): raw image context
        boxes (array): array of result file
            Shape: [num_detections, 4] for BB output / [num_detections, 4]
            for QUAD output
    Return:
        None
    """
    image = np.array(image)

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(image_path))

    # result directory
    res_file = output_dir + "res_" + filename + '.txt'
    res_img_file = output_dir + "res_" + filename + '.jpg'

    with open(res_file, 'w') as f:
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            strResult = ','.join([str(p) for p in poly]) + '\r\n'
            f.write(strResult)

            poly = poly.reshape(-1, 2)
            cv2.polylines(image,
                          [poly.reshape((-1, 1, 2))],
                          True,
                          color=(0, 0, 255),
                          thickness=2)
#            ptColor = (0, 255, 255)
#            if verticals is not None:
#                if verticals[i]:
#                    ptColor = (255, 0, 0)

            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(image, "{}".format(texts[i]),
                            (poly[0][0]+1, poly[0][1]+1),
                            font,
                            font_scale,
                            (0, 0, 0),
                            thickness=1)
                cv2.putText(image,
                            "{}".format(texts[i]),
                            tuple(poly[0]),
                            font,
                            font_scale,
                            (0, 255, 255),
                            thickness=1)

    # Save result image
    cv2.imwrite(res_img_file, image)
