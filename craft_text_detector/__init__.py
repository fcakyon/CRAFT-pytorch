from __future__ import absolute_import

__version__ = "0.1.1"

from craft_text_detector.imgproc import read_image

from craft_text_detector.file_utils import (export_detected_regions,
                                            export_extra_results)

from craft_text_detector.predict import (load_craftnet_model,
                                         load_refinenet_model,
                                         get_prediction)

# load craft model
craft_net = load_craftnet_model()


# detect texts
def detect_text(image_path,
                output_dir=None,
                export_extra=True,
                text_threshold=0.7,
                link_threshold=0.4,
                low_text=0.4,
                cuda=False,
                show_time=False,
                refiner=False):
    """
    Arguments:
        image_path: path to the image to be processed
        output_dir: path to the results to be exported
        export_extra: export heatmap, detection points, box visualization
        text_threshold: text confidence threshold
        link_threshold: link confidence threshold
        low_text: text low-bound score
        cuda: Use cuda for inference
        poly: enable polygon type
        show_time: show processing time
        refiner: enable link refiner
    """
    # load image
    image = read_image(image_path)

    # load refiner if required
    if refiner:
        refine_net = load_refinenet_model()
    else:
        refine_net = None

    # perform prediction
    bboxes, polys, heatmap = get_prediction(image=image,
                                            craft_net=craft_net,
                                            refine_net=refine_net,
                                            text_threshold=text_threshold,
                                            link_threshold=link_threshold,
                                            low_text=low_text,
                                            cuda=cuda,
                                            show_time=show_time)

    # export if output_dir is given
    if output_dir is not None:
        # export detected text regions
        export_detected_regions(image_path=image_path,
                                image=image,
                                regions=polys,
                                output_dir=output_dir)

        # export heatmap, detection points, box visualization
        if export_extra:
            export_extra_results(image_path=image_path,
                                 image=image,
                                 regions=polys,
                                 heatmap=heatmap,
                                 output_dir=output_dir)

    # return prediction results
    return bboxes, polys, heatmap
