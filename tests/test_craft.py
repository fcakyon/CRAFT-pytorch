import os
import unittest
import craft_text_detector


class TestCraftTextDetector(unittest.TestCase):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(root_dir, '../figures/idcard.png')

    def test_load_craftnet_model(self):
        craft_net = craft_text_detector.load_craftnet_model()
        self.assertTrue(craft_net)

    def test_load_refinenet_model(self):
        refine_net = craft_text_detector.load_refinenet_model()
        self.assertTrue(refine_net)

    def test_get_prediction(self):
        # load image
        image = craft_text_detector.read_image(self.image_path)

        # load models
        craft_net = craft_text_detector.load_craftnet_model()
        refine_net = None

        # perform prediction
        text_threshold = 0.9
        link_threshold = 0.2
        low_text = 0.2
        cuda = False
        show_time = False
        get_prediction = craft_text_detector.get_prediction
        prediction_result = get_prediction(image=image,
                                           craft_net=craft_net,
                                           refine_net=refine_net,
                                           text_threshold=text_threshold,
                                           link_threshold=link_threshold,
                                           low_text=low_text,
                                           cuda=cuda,
                                           show_time=show_time)
        self.assertEqual(len(prediction_result["boxes"]), 49)
        self.assertEqual(len(prediction_result["boxes"][0]), 4)
        self.assertEqual(len(prediction_result["boxes"][0][0]), 2)
        self.assertEqual(int(prediction_result["boxes"][0][0][0]), 112)
        self.assertEqual(len(prediction_result["polys"]), 49)
        self.assertEqual(prediction_result["heatmap"].shape, (384, 1184, 3))

    def test_detect_text(self):
        prediction_result = craft_text_detector.detect_text(image_path=self.image_path,
                                                            output_dir=None,
                                                            rectify=True,
                                                            export_extra=False,
                                                            text_threshold=0.7,
                                                            link_threshold=0.4,
                                                            low_text=0.4,
                                                            cuda=False,
                                                            show_time=False,
                                                            refiner=False,
                                                            crop_type="poly")
        self.assertEqual(len(prediction_result["boxes"]), 56)
        self.assertEqual(len(prediction_result["boxes"][0]), 4)
        self.assertEqual(len(prediction_result["boxes"][0][0]), 2)
        self.assertEqual(int(prediction_result["boxes"][0][0][0]), 114)

        prediction_result = craft_text_detector.detect_text(image_path=self.image_path,
                                                            output_dir=None,
                                                            rectify=True,
                                                            export_extra=False,
                                                            text_threshold=0.7,
                                                            link_threshold=0.4,
                                                            low_text=0.4,
                                                            cuda=False,
                                                            show_time=False,
                                                            refiner=True,
                                                            crop_type="poly")
        self.assertEqual(len(prediction_result["boxes"]), 19)
        self.assertEqual(len(prediction_result["boxes"][0]), 4)
        self.assertEqual(len(prediction_result["boxes"][0][0]), 2)
        self.assertEqual(int(prediction_result["boxes"][0][2][0]), 661)


        prediction_result = craft_text_detector.detect_text(image_path=self.image_path,
                                                            output_dir=None,
                                                            rectify=False,
                                                            export_extra=False,
                                                            text_threshold=0.7,
                                                            link_threshold=0.4,
                                                            low_text=0.4,
                                                            cuda=False,
                                                            show_time=False,
                                                            refiner=False,
                                                            crop_type="box")
        self.assertEqual(len(prediction_result["boxes"]), 56)
        self.assertEqual(len(prediction_result["boxes"][0]), 4)
        self.assertEqual(len(prediction_result["boxes"][0][0]), 2)
        self.assertEqual(int(prediction_result["boxes"][0][2][0]), 242)

        prediction_result = craft_text_detector.detect_text(image_path=self.image_path,
                                                            output_dir=None,
                                                            rectify=False,
                                                            export_extra=False,
                                                            text_threshold=0.7,
                                                            link_threshold=0.4,
                                                            low_text=0.4,
                                                            cuda=False,
                                                            show_time=False,
                                                            refiner=True,
                                                            crop_type="box")
        self.assertEqual(len(prediction_result["boxes"]), 19)
        self.assertEqual(len(prediction_result["boxes"][0]), 4)
        self.assertEqual(len(prediction_result["boxes"][0][0]), 2)
        self.assertEqual(int(prediction_result["boxes"][0][2][0]), 661)


if __name__ == '__main__':
    unittest.main()
