from ..joblib.subprocessbase import SubprocessorBase
from face_alignment import FaceAlignment, LandmarksType
import cv2
import numpy as np
from ..landmarkprocess import landmarkprocessor, get_image_hull_mask
from ..landmarkprocess.landmarkprocessor import FaceType


class Converter(object):
    def __init__(self, predictor=None, predictor_input_size=(256, 256), predictor_output_size=(256, 256),
                 transformed_fixed_size=(256, 256), device="cpu"):
        self.predictor = predictor
        self.predictor_input_size = predictor_input_size
        self.device = device
        self.predictor_output_size = predictor_output_size
        self.transformed_fixed_size = transformed_fixed_size
        self.on_cli_initialize()

    # overridable
    def on_cli_initialize(self):
        self.fa = FaceAlignment(
            LandmarksType._2D, device=self.device)

    def cli_convert_face(self, image_rgb):
        '''
        image_rgb is the image of user
        '''
        landmark_user = self.fa.get_landmarks(image_rgb)[-1]
        print(landmark_user.shape)
        mask_user = get_image_hull_mask(np.shape(image_rgb), landmark_user)

        # get the transform matrix
        face_mat = landmarkprocessor.get_transform_mat(
            landmark_user, self.transformed_fixed_size[0], face_type=FaceType.FULL)
        face_output_mat = landmarkprocessor.get_transform_mat(
            # , scale=0.9
            landmark_user, self.transformed_fixed_size[0], face_type=FaceType.FULL, scale=1.1
        )

        # transform the face to appropriate position (forward-looking)
        frontface_image = cv2.warpAffine(
            image_rgb, face_mat, self.transformed_fixed_size, flags=cv2.INTER_LANCZOS4)
        frontface_mask = cv2.warpAffine(
            mask_user, face_mat, self.transformed_fixed_size, flags=cv2.INTER_LANCZOS4)

        # ---------------------------------prepare ppt data-----------------------------------------
        # cv2.imshow("original image", cv2.cvtColor(
        #     image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
        landmark_image = image_rgb.astype(np.uint8).copy()
        for point in landmark_user:
            cv2.circle(landmark_image, (
                int(point[0]), int(point[1])), 10, (255, 0, 0))
        cv2.imshow("landmarkImage", cv2.cvtColor(
            landmark_image, cv2.COLOR_RGB2BGR))
        cv2.imshow("frontface_image", cv2.cvtColor(
            frontface_image, cv2.COLOR_RGB2BGR))
        cv2.imshow("frontface_mask", (frontface_mask*255).astype(np.uint8))
        cv2.imshow("mask_user", (mask_user*255).astype(np.uint8))
        cv2.waitKey(1)

        # ------------------------------------------------------------------------------------------

        input_predictor = cv2.resize(
            frontface_image, self.predictor_input_size)

        # generate a new swapped face and reshape it into appropriate size,then transform it into image_rgb's position
        face_generate = self.predictor(
            cv2.cvtColor(input_predictor, cv2.COLOR_RGB2BGR))
        # -------------------------------------------------------------------
        # landmark_face_generate = self.fa.get_landmarks(face_generate)[-1]
        # transform_generate_mat = landmarkprocessor.get_transform_mat(
        #     landmark_face_generate, self.transformed_fixed_size[0], face_type=FaceType.FULL)
        # face_generate = cv2.warpAffine(
        #     face_generate, transform_generate_mat, self.transformed_fixed_size, flags=cv2.INTER_LANCZOS4)
        # -------------------------------------------------------------------
        frontface_generator = cv2.resize(
            face_generate, self.transformed_fixed_size)
        swapped_face_image = cv2.warpAffine(frontface_generator, face_output_mat, (image_rgb.shape[1], image_rgb.shape[0]), np.zeros(
            image_rgb.shape, dtype=np.float32), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4).astype(np.uint8)

        cv2.imshow("face_generate", cv2.cvtColor(
            face_generate.astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imshow("swapped_face_image", cv2.cvtColor(
            swapped_face_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        # calculate the final  mask of image_rgb
        mask = frontface_mask[..., np.newaxis]
        mask_rgb = np.repeat(mask, 3, axis=-1)
        mask_image_rgb = cv2.warpAffine(mask_rgb, face_output_mat, (image_rgb.shape[1], image_rgb.shape[0]), np.zeros(
            image_rgb.shape, dtype=np.float32), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4)

        out_img = image_rgb.copy().astype(np.uint8)
        # for i in range(np.shape(image_rgb)[0]):
        #     for j in range(np.shape(image_rgb)[1]):
        #         if mask_image_rgb[i, j, 0] != 0:
        #             out_img[i, j, :] = swapped_face_image[i, j, :]

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
            swapped_face_image, out_img, (img_face_seamless_mask_a*255).astype(np.uint8), (s_maskx, s_masky), cv2.NORMAL_CLONE)
        # ------------------------------------------------------------------------------------------
        return seamless_out

    def __call__(self, image_rgb):
        return self.cli_convert_face(image_rgb)
