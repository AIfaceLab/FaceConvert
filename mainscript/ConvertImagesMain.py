# from packages.converters.Converter_Process import ConverterProcess
from packages.utils.path_utils import *
import cv2
from Modules import PredicterBase
from packages.converters.ConvertImages import ConvertImages


def main(args):
    input_images_dir = args.get('input_images_dir', None)
    input_faces_dir = args.get('input_faces_dir', None)
    output_dir = args.get('output_dir', None)
    print("converter main function")
    print("input_images_dir_src is :", input_images_dir)
    print("input_faces_dir_src is :", input_faces_dir)
    print("output_dir is :", output_dir)

    image_paths = get_image_paths(input_images_dir)
    face_paths = get_image_paths(input_faces_dir)
    converter = ConvertImages(image_paths, face_paths, face_fixed_size=(
        256, 256), output_dir=output_dir, workers=1, converter_device="cuda:0")
    converter.run()
