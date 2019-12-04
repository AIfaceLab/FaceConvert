from ..joblib.subprocessbase import SubprocessorBase
from face_alignment import FaceAlignment, LandmarksType
import cv2
import numpy as np
from ..landmarkprocess import landmarkprocessor, get_image_hull_mask
from ..landmarkprocess.landmarkprocessor import FaceType
from .converter import Converter
from Modules import PredicterBase, PredicterDeepfakes
import time
from pathlib import Path
import os


class ConverterProcess(SubprocessorBase):
    class Cli(SubprocessorBase.Cli):

        # overriable
        def process_data(self, data):
            idx, image_path_dict = data
            image = cv2.imread(image_path_dict['input_image_path'])
            # image = cv2.resize(image, (1920, 1080))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out = self.converter(image)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            print(image_path_dict['input_image_path'],
                  Path(image_path_dict['input_image_path']).name)
            out_path = os.path.join(image_path_dict['output_dir'], Path(
                image_path_dict['input_image_path']).name)
            cv2.imshow("result", out.astype(np.uint8))
            cv2.imwrite(out_path, out.astype(np.uint8))
            cv2.waitKey(1)

        def on_cli_initialize(self, client_dict):
            # predictor = PredicterBase(
            #     model_parameters_name=client_dict['predictor_model_name'], device=client_dict['predictor_device'])
            predictor = PredicterDeepfakes(input_size=client_dict['predictor_input_size'],
                                           device=client_dict['predictor_device'], model_parameters_name=client_dict['predictor_model_name'])
            self.converter = Converter(predictor, predictor_input_size=client_dict['predictor_input_size'],
                                       transformed_fixed_size=client_dict['face_fixed_size'], device=client_dict['converter_device'])
            # time.sleep(5)

    def __init__(self, img_path_list, predictor=None, predictor_input_size=(256, 256),
                 face_fixed_size=(256, 256), output_dir='results', workers=1, predictor_device="cuda", converter_device="cuda"):
        '''
            src_image_path_list, the face which will be the output \n
            [obj_img_path_list]  the actor's faces\n
            predictor  the neural network which will output a face like src-images by inputting obj image\n
            [predictor_input_size] is the input shape of neural network\n
            [face_fixed_size] is the shape of transformed Forward-looking image which is used to calculate tranformat matrix
            [workers] is the number of threads \n
            [device] is the device where the neural network is
        '''
        self.imgs = img_path_list
        self.input_data_idxs = [*range(len(img_path_list))]
        self.num_subprocessers = workers
        self.predictor_device = predictor_device
        self.converter_device = converter_device
        self.predictor = predictor
        self.predictor_input_size = predictor_input_size
        self.face_fixed_size = face_fixed_size
        # output dir
        self.output_dir = output_dir
        if not Path(self.output_dir).exists():
            os.mkdir(self.output_dir)
        # 60 is the no response time
        super().__init__('Converter', ConverterProcess.Cli, 60)
        pass

    # overriable
    def process_convert(self):
        raise NotImplementedError

    # overridable
    def preprocess(self):
        raise NotImplementedError

    # postprocess
    def postprocess(self):
        raise NotImplementedError

    # override
    # host_dict is useless for this class
    def get_data(self, host_dict):
        if len(self.input_data_idxs) > 0:
            idx = self.input_data_idxs.pop(0)
            return (idx, {"input_image_path": self.imgs[idx], "output_dir": self.output_dir})
        return None

    def process_info_generator(self):
        for i in range(self.num_subprocessers):
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i),
                                      'predictor_device': self.predictor_device,
                                      'predictor_model_name': self.predictor,
                                      'predictor_input_size': self.predictor_input_size,
                                      'face_fixed_size': self.face_fixed_size,
                                      'converter_device': self.converter_device,
                                      'output_dir': self.output_dir
                                      }
