[![PyPI version](https://badge.fury.io/py/craft-text-detector.svg)](https://badge.fury.io/py/craft-text-detector)
[![Conda version](https://anaconda.org/fcakyon/craft-text-detector/badges/version.svg)](https://anaconda.org/fcakyon/craft-text-detector)
[![CI](https://github.com/fcakyon/craft-text-detector/workflows/CI/badge.svg)](https://github.com/fcakyon/craft-text-detector/actions?query=event%3Apush+branch%3Amaster+is%3Acompleted+workflow%3ACI)


## CRAFT: Character-Region Awareness For Text detection
Packaged Version of the Official Pytorch implementation of CRAFT text detector | [Paper](https://arxiv.org/abs/1904.01941) |  [Supplementary](https://youtu.be/HI8MzpY8KMI) |

**[Youngmin Baek](mailto:youngmin.baek@navercorp.com), Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee.**

 **Package maintainer: Fatih Cagatay Akyon**

### Overview
PyTorch implementation for CRAFT text detector that effectively detect text area by exploring each character region and affinity between characters. The bounding box of texts are obtained by simply finding minimum bounding rectangles on binary map after thresholding character region and affinity scores.

<img width="1000" alt="teaser" src="./figures/craft_example.gif">

## Updates
**6 April, 2020**: Conda package release

**1 April, 2020**: Python 3.8 support, removed skimage dependency

**24 March, 2020**: Polygon rectification support

**23 March, 2020**: Python 3.5 support

**21 March, 2020**: Initial package release


## Getting started
### Installation
- Install using conda for Linux, Mac and Windows (preferred):
```console
conda install -c fcakyon craft-text-detector
```
- Install using pip for Linux and Mac:
```console
pip install craft-text-detector
```

### Basic Usage
```python
# import package
import craft_text_detector as craft

# set image path and export folder directory
image_path = 'figures/idcard.png'
output_dir = 'outputs/'

# apply craft text detection and export detected regions to output directory
prediction_result = craft.detect_text(image_path, output_dir, crop_type="poly", cuda=False)
```

### Advanced Usage
```python
# import package
import craft_text_detector as craft

# set image path and export folder directory
image_path = 'figures/idcard.png'
output_dir = 'outputs/'

# read image
image = craft.read_image(image_path)

# load models
refine_net = craft.load_refinenet_model()
craft_net = craft.load_craftnet_model()

# perform prediction
prediction_result = craft.get_prediction(image=image,
				         craft_net=craft_net,
				         refine_net=refine_net,
				         text_threshold=0.7,
				         link_threshold=0.4,
				         low_text=0.4,
				         cuda=True,
				         mag_ratio=0.8,
				         show_time=True)

# export detected text regions
exported_file_paths = craft.export_detected_regions(image_path=image_path,
                                                    image=image,
                                                    regions=prediction_result["boxes"],
                                                    output_dir=output_dir,
                                                    rectify=True)

# export heatmap, detection points, box visualization
craft.export_extra_results(image_path=image_path,
    	                   image=image,
                           regions=prediction_result["boxes"],
                           heatmap=prediction_result["heatmap"],
                           output_dir=output_dir)
```

