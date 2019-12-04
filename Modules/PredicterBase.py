
from .base.model_DF import Encoder, Decoder
import torch
import torch.nn as nn
from pathlib import Path
import os


class PredicterBase(object):
    def __init__(self, input_size=(256, 256), output_size=(256, 256),
                 model_parameters_name="Basic_Autoencoder.tar", device="cpu"):
        super().__init__()
        assert isinstance(model_parameters_name,
                          str), "model_parameters must be a string "
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.model_parameters_path = os.path.join(
            Path(__file__).parent, Path(model_parameters_name))
        self.create_model()

    def create_model(self):
        class Model(nn.Module):
            def __init__(self, input_size, output_size):
                super().__init__()
                self.encoder = Encoder(shape=input_size[0])
                self.decoder = Decoder(shape=output_size[0])
                # self.encoder = Encoder()
                # self.decoder = Decoder()

            def forward(self, image):
                out = self.encoder(image)
                out = self.decoder(out)
                return out
        self.model = Model(self.input_size, self.output_size).to(self.device)
        self.torch_model_parameters = torch.load(self.model_parameters_path)
        self.model.encoder.load_state_dict(
            self.torch_model_parameters['Encoder_state_dict'])
        self.model.decoder.load_state_dict(
            self.torch_model_parameters['DecoderA_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        '''
            image must be a numpy.array with three(RGB) channels
        '''
        image_tensor = torch.from_numpy(image).type(dtype=torch.float)
        image_tensor = image_tensor.transpose(0, 2).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)
        # if self.device != "cpu":
        image_tensor.to(self.device)
        swapface_abgr = self.model(image_tensor)
        swapface_bgr = swapface_abgr[0][:3, :, :]

        out = swapface_bgr.transpose(0, 2).cpu().detach().numpy()
        return out

    def __call__(self, image):
        return self.predict(image)
