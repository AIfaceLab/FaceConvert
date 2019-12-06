from ..joblib.subprocessbase import SubprocessorBase
from face_alignment import FaceAlignment, LandmarksType
import cv2
import numpy as np
from ..landmarkprocess import landmarkprocessor, get_image_hull_mask
from ..landmarkprocess.landmarkprocessor import FaceType
from Modules import PredicterBase, PredicterDeepfakes
import time
from pathlib import Path
import os
from face_compare import FaceCompare


class ConvertImages(SubprocessorBase):
    class Cli(SubprocessorBase.Cli):

        class Converter(object):
            def __init__(self, transformed_fixed_size=(256, 256), standard_face=None, scale=1.0, device="cuda"):
                super().__init__()
                self.transformed_fixed_size = transformed_fixed_size
                self.device = device
                self.standard_face = standard_face
                self.scale = scale
                self.on_initialize()

            def on_initialize(self):
                self.fa = FaceAlignment(
                    LandmarksType._2D, device=self.device)
                self.facecompare = FaceCompare(device=self.device)

            # def face_compare(self, face_rgb_list):
            #     return face_rgb_list[0]

            def preprocess_image(self, image_rgb, landmark, face):
                h, w = np.shape(face)[:2]
                # assert h == w, "the width and height of face must be equal"

                mask = get_image_hull_mask(np.shape(image_rgb), landmark)
                # get the transform matrix
                face_mat = landmarkprocessor.get_transform_mat(
                    landmark, h, face_type=FaceType.FULL)
                face_output_mat = landmarkprocessor.get_transform_mat(
                    landmark, h, face_type=FaceType.FULL, scale=self.scale
                )
                # transform the face to appropriate position (forward-looking)
                frontface_image = cv2.warpAffine(
                    image_rgb, face_mat, (h, w), flags=cv2.INTER_LANCZOS4)
                frontface_mask = cv2.warpAffine(
                    mask, face_mat, (h, w), flags=cv2.INTER_LANCZOS4)

                # cv2.imshow("src_face", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                # cv2.imshow("obj_face", cv2.cvtColor(
                #     frontface_image, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(1)

                # face ,the shape is same as image_rgb
                swapped_face_image = cv2.warpAffine(face, face_output_mat, (image_rgb.shape[1], image_rgb.shape[0]), np.zeros(
                    image_rgb.shape, dtype=np.float32), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4).astype(np.uint8)
                # mask
                mask_n = frontface_mask[..., np.newaxis]
                mask_rgb = np.repeat(mask_n, 3, axis=-1)
                mask_image_rgb = cv2.warpAffine(mask_rgb, face_output_mat, (image_rgb.shape[1], image_rgb.shape[0]), np.zeros(
                    image_rgb.shape, dtype=np.float32), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4)

                out_dict = {}
                out_dict["image_rgb"] = image_rgb
                out_dict["image_mask"] = mask_image_rgb
                out_dict["image_face"] = swapped_face_image
                out_dict["compare_face"] = frontface_image
                return out_dict

            def postprocess(self, image_rgb, mask_image_rgb, swapped_face_image):
                # -----------------------------seamless clone-----------------------------------------------
                img_face_mask_a = mask_image_rgb[..., 0:1]
                img_face_seamless_mask_a = None
                # convert mask to binary(0 or 255)
                for i in range(1, 10):
                    a = img_face_mask_a > i / 10.0
                    if len(np.argwhere(a)) == 0:
                        continue
                    img_face_seamless_mask_a = img_face_mask_a.copy()
                    img_face_seamless_mask_a[a] = 1.0
                    img_face_seamless_mask_a[img_face_seamless_mask_a <=
                                             i / 10.0] = 0.0
                    break
                # cv2.imshow("seamless mask", img_face_seamless_mask_a)
                # cv2.waitKey(0)
                l, t, w, h = cv2.boundingRect(
                    (img_face_seamless_mask_a*255).astype(np.uint8))
                s_maskx, s_masky = int(l+w/2), int(t+h/2)
                seamless_out = cv2.seamlessClone(
                    swapped_face_image, image_rgb, (img_face_seamless_mask_a*255).astype(np.uint8), (s_maskx, s_masky), cv2.NORMAL_CLONE)
                # ------------------------------------------------------------------------------------------
                return seamless_out

            def face_compare(self, out_list):
                # the face must be rgb and (0~255)
                scores_list = []
                for out_dict in out_list:
                    scores_list.append(self.facecompare(
                        out_dict['compare_face'], self.standard_face))
                # find the suitable faces
                suitable_faces_preprocess_out = [
                    out_list[i] for i in self._find_compared_face_strategy(scores_list)]
                return suitable_faces_preprocess_out

            def convert_face(self, image_rgb, face):
                landmarks = self.fa.get_landmarks(image_rgb)
                print(len(landmarks))
                # get all faces in the image and put them in a list
                preprocess_out_list = []
                for landmark in landmarks:
                    out_dict = self.preprocess_image(
                        image_rgb, landmark, face)
                    preprocess_out_list.append(out_dict)
                # select the suitable faces
                suitable_preprocess_out = self.face_compare(
                    preprocess_out_list)
                result_image = self._processe_suitable_out(
                    suitable_preprocess_out)
                if result_image is not None:
                    return result_image
                else:
                    return image_rgb

            def _processe_suitable_out(self, input_):
                if len(input_) == 0:
                    '''no face suitable'''
                    print("no face detected")
                    return None
                if len(input_) == 1:
                    return self.postprocess(input_[0]['image_rgb'], input_[0]["image_mask"], input_[0]["image_face"])
                else:
                    temp_dict = input_[0]
                    postprocessed_out = self.postprocess(
                        temp_dict["image_rgb"], temp_dict["image_mask"], temp_dict["image_face"])
                    next_input = input_[1:]
                    next_input[0]["image_rgb"] = postprocessed_out
                    return self._processe_suitable_out()

            def _find_compared_face_strategy(self, scores_list, threshold=0.74):
                '''
                you should design greater strategy
                '''
                # threshold_results = [
                #     x if x >= threshold else 0 for x in scores_list]
                print("scores_list :", scores_list)
                if max(scores_list) > threshold:
                    return [scores_list.index(max(scores_list))]
                else:
                    return []

            def __call__(self, image_rgb, face):
                return self.convert_face(image_rgb, face)
        # overriable

        def process_data(self, data):
            idx, image_path_dict = data
            image = cv2.imread(image_path_dict['input_image_path'])
            face = cv2.imread(image_path_dict['input_face'])

            image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
            face = cv2.resize(face, (0, 0), fx=0.25, fy=0.25)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            out = self.converter(image, face)

            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(image_path_dict['output_dir'], Path(
                image_path_dict['input_image_path']).name)
            cv2.imshow("result", cv2.resize(
                out.astype(np.uint8), (0, 0), fx=0.25, fy=0.25))
            cv2.imwrite(out_path, out.astype(np.uint8))
            cv2.waitKey(1)

        def on_cli_initialize(self, client_dict):
            # --------------------------------------------------------------------------------------
            # 今天要写这里
            standard_face = cv2.imread(client_dict['standard_face_path'][0])
            standard_face = cv2.cvtColor(standard_face, cv2.COLOR_BGR2RGB)
            # --------------------------------------------------------------------------------------
            self.converter = ConvertImages.Cli.Converter(standard_face=standard_face,
                                                         transformed_fixed_size=client_dict['face_fixed_size'], device=client_dict['converter_device'])
            # time.sleep(5)

    def __init__(self, img_path_list, face_path_list, standardface_path_list, face_fixed_size=(256, 256), output_dir='results', workers=1, converter_device="cuda"):
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
        self.faces = face_path_list
        self.standard_face = standardface_path_list
        self.input_data_idxs = [*range(len(img_path_list))]
        self.num_subprocessers = workers
        self.converter_device = converter_device
        self.face_fixed_size = face_fixed_size
        # output dir
        self.output_dir = output_dir
        if not Path(self.output_dir).exists():
            os.mkdir(self.output_dir)
        # 60 is the no response time
        super().__init__('Converter', ConvertImages.Cli, 60)

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
            return (idx, {"input_image_path": self.imgs[idx], "input_face": self.faces[idx], "output_dir": self.output_dir})
        return None

    def process_info_generator(self):
        for i in range(self.num_subprocessers):
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i),
                                      'face_fixed_size': self.face_fixed_size,
                                      'converter_device': self.converter_device,
                                      'output_dir': self.output_dir,
                                      'standard_face_path': self.standard_face
                                      }
