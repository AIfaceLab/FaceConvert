# from packages.converters.Converter_Process import ConverterProcess
from packages.utils.path_utils import *
import cv2
from Modules import PredicterBase
from packages.converters.ConverterProcess import ConverterProcess


def main(args):
    input_dir = args.get('input_dir', None)
    output_dir = args.get('output_dir', None)
    print("converter main function")
    print("input_dir_src is :", input_dir)
    print("output_dir is :", output_dir)

    image_paths = get_image_paths(input_dir)
    converter = ConverterProcess(
        image_paths, predictor="model_256.pth", predictor_input_size=(256, 256), face_fixed_size=(256, 256), output_dir=output_dir,
        workers=1, predictor_device="cuda:1", converter_device="cuda:0")
    converter.run()
