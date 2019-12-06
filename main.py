from joblib import *
import cv2
import argparse
import os
# from mainscript.ConverterMain import main
import mainscript


class fixPathAction(argparse.Action):
    # used to fix the string of args to Path
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(
            os.path.expanduser(values)))


class testaction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(
            os.path.expanduser(values)))
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # ----------------------------------------------------------------------------------------------
    # convert:image matting
    def ProcessConvert(arguments):
        print("-------------------------------------------------\n",
              "****Process the image with naive ways****")
        args = {
            'input_dir': arguments.input_dir_actor,
            'output_dir': arguments.output_dir
        }

        mainscript.ConverterMain.main(args)
        pass

    p = subparsers.add_parser(
        "naivecopy", help="just directly copy the src to obj")
    p.add_argument("--input-dir-actor", required=True, action=fixPathAction)
    p.add_argument("--output-dir", required=True, action=fixPathAction)
    p.set_defaults(function=ProcessConvert)

    # ----------------------------------------------------------------------------------------------
    # convert faces with two image directories
    def ProcessConvertImages(arguments):
        print("-------------------------------------------------\n",
              "****Process convert images****")
        args = {
            'input_images_dir': arguments.input_dir_actor,
            'input_faces_dir': arguments.input_dir_face,
            'output_dir': arguments.output_dir,
            'standard_face_dir': arguments.standard_face_dir
        }
        mainscript.ConvertImagesMain.main(args)
        pass
    p = subparsers.add_parser(
        "convertimages", help="convert images with face images")
    p.add_argument("--input-dir-actor", required=True, action=fixPathAction)
    p.add_argument("--input-dir-face", required=True, action=fixPathAction)
    p.add_argument("--output-dir", required=True, action=fixPathAction)
    p.add_argument("--standard-face-dir", required=True, action=fixPathAction)
    p.set_defaults(function=ProcessConvertImages)
    # ----------------------------------------------------------------------------------------------

    args = parser.parse_args()
    args.function(args)
