from .deepfakes import Encoder, Decoder
import torch
import torch.nn as nn
from pathlib import Path
import os
from .PredicterBase import PredicterBase
import cv2
import numpy as np
from packages.mathlib.umeyama import umeyama


class PredicterDeepfakes(PredicterBase):
    def __init__(self, input_size=(128, 128), output_size=(128, 128), model_parameters_name='Basic_Autoencoder.tar', device='cpu'):
        super().__init__(input_size=input_size, output_size=output_size,
                         model_parameters_name=model_parameters_name, device=device)

    def create_model(self):
        class Model(nn.Module):
            def __init__(self, input_size=(128, 128), output_size=(128, 128)):
                super().__init__()
                self.encoder = Encoder(shape=input_size[0])
                self.decoder = Decoder(shape=input_size[0])

            def forward(self, image):
                out = self.encoder(image)
                rgb, mask = self.decoder(out)
                return rgb, mask
        self.model = Model(self.input_size, self.input_size).to(self.device)
        self.torch_model_parameters = torch.load(self.model_parameters_path)
        self.model.encoder.load_state_dict(
            self.torch_model_parameters['Encoder_state_dict'])
        self.model.decoder.load_state_dict(
            self.torch_model_parameters['DecoderA_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        # image /= 255
        warped_image, target_image = self.random_warp_rev(
            image, self.input_size[0])
        image_tensor = torch.from_numpy(image).type(dtype=torch.float)
        image_tensor /= 255
        image_tensor = image_tensor.transpose(0, 2).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)
        # if self.device != "cpu":
        image_tensor.to(self.device)
        rgb, mask = self.model(image_tensor)
        out = rgb[0].transpose(0, 2).cpu().detach().numpy()*255
        # ------------------------------------------------------------------------
        # out = rgb[0].cpu().detach().numpy()*255
        # b, g, r = out
        # b, g, r = b[:][np.newaxis], g[:][np.newaxis], r[:][np.newaxis]
        # out = np.stack((b, g, r), axis=-1)[0]
        # ------------------------------------------------------------------------
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        return out

    def __call__(self, image):
        return super().__call__(image)

    def random_warp_rev(self, image, res):
        image = cv2.resize(image, (256, 256))
        res_scale = 256//64
        assert res_scale >= 1, f"Resolution should be >= 64. Recieved {res}."

        interp_param = 80 * res_scale
        interp_slice = slice(interp_param//10, 9*interp_param//10)
        dst_pnts_slice = slice(0, 65*res_scale, 16*res_scale)

        rand_coverage = 256/2  # random warping coverage
        rand_scale = np.random.uniform(5., 6.2)  # random warping scale

        range_ = np.linspace(128-rand_coverage, 128+rand_coverage, 5)
        mapx = np.broadcast_to(range_, (5, 5))
        mapy = mapx.T
        mapx = mapx + np.random.normal(size=(5, 5), scale=rand_scale)
        mapy = mapy + np.random.normal(size=(5, 5), scale=rand_scale)
        interp_mapx = cv2.resize(mapx, (interp_param, interp_param))[
            interp_slice, interp_slice].astype('float32')
        interp_mapy = cv2.resize(mapy, (interp_param, interp_param))[
            interp_slice, interp_slice].astype('float32')
        warped_image = cv2.remap(
            image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
        src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
        dst_points = np.mgrid[dst_pnts_slice, dst_pnts_slice].T.reshape(-1, 2)
        mat = umeyama(src_points, dst_points, True)[0:2]
        target_image = cv2.warpAffine(image, mat, (256, 256))

        warped_image = cv2.resize(warped_image, (res, res))
        target_image = cv2.resize(target_image, (res, res))
        return warped_image, target_image
