import torch
import torch.nn as nn
import numpy as np
import imp
import torchvision
from torchvision.models import vgg19
from .Block_2 import Cropped_VGG19

def distance(loss='l1'):
    if loss == 'l1':
        return nn.L1Loss()
    elif loss == 'l2':
        return nn.MSELoss()
    elif loss == 'Cross':
        return nn.CrossEntropyLoss()
    else:
        print('NO SUCH LOSS !')


class cyclic_loss(nn.Module):
    def __init__(self):
        super(cyclic_loss, self).__init__()

    def forward(self, E, G1, G2, real1, criterion):
        fake_2_BGRA = G2(E(real1))
        fake_2_BGR = fake_2_BGRA[:, :3, :, :]
        fake_2_mask = fake_2_BGRA[:, 3:, :, :]

        cyclic_1_BGRA = G1(E(fake_2_BGR))
        cyclic_1_BGR = cyclic_1_BGRA[:, :3, :, :]
        cyclic_1_mask = cyclic_1_BGRA[:, 3:, :, :]

        loss = criterion(cyclic_1_BGR, real1)
        loss += 0*criterion(cyclic_1_mask, fake_2_mask)
        return loss





class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class LossCnt(nn.Module):
    def __init__(self, VGGFace_body_path, VGGFace_weight_path, device):
        super(LossCnt, self).__init__()

        self.VGG19 = vgg19(pretrained=True)
        self.VGG19.eval()
        self.VGG19.to(device)

        MainModel = imp.load_source('MainModel', VGGFace_body_path)
        full_VGGFace = torch.load(VGGFace_weight_path, map_location='cpu')
        cropped_VGGFace = Cropped_VGG19()
        #        cropped_VGGFace.load_state_dict(full_VGGFace.state_dict(), strict = False)
        cropped_VGGFace.load_state_dict(full_VGGFace, strict=False)
        self.VGGFace = cropped_VGGFace
        self.VGGFace.eval()
        self.VGGFace.to(device)

    def forward(self, x, x_hat, vgg19_weight=1e-2, vggface_weight=2e-3):
        l1_loss = nn.L1Loss()

        """Retrieve vggface feature maps"""
        with torch.no_grad():  # no need for gradient compute
            vgg_x_features = self.VGGFace(x)  # returns a list of feature maps at desired layers

        vgg_xhat_features = self.VGGFace(x_hat)

        lossface = 0
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            lossface += l1_loss(x_feat, xhat_feat)

        """Retrieve vggface feature maps"""

        # define hook
        def vgg_x_hook(module, input, output):
            output.detach_()  # no gradient compute
            vgg_x_features.append(output)

        def vgg_xhat_hook(module, input, output):
            vgg_xhat_features.append(output)

        vgg_x_features = []
        vgg_xhat_features = []

        vgg_x_handles = []

        conv_idx_list = [2, 7, 12, 21, 30]  # idxes of conv layers in VGG19 cf.paper
        conv_idx_iter = 0

        # place hooks
        for i, m in enumerate(self.VGG19.features.modules()):
            if i == conv_idx_list[conv_idx_iter]:
                if conv_idx_iter < len(conv_idx_list) - 1:
                    conv_idx_iter += 1
                vgg_x_handles.append(m.register_forward_hook(vgg_x_hook))

        # run model for x
        self.VGG19(x)

        # retrieve features for x
        for h in vgg_x_handles:
            h.remove()

        # retrieve features for x_hat
        conv_idx_iter = 0
        for i, m in enumerate(self.VGG19.modules()):
            if i <= 30:  # 30 is last conv layer
                if type(m) is not torch.nn.Sequential and type(m) is not torchvision.models.vgg.VGG:
                    # only pass through nn.module layers
                    if i == conv_idx_list[conv_idx_iter]:
                        if conv_idx_iter < len(conv_idx_list) - 1:
                            conv_idx_iter += 1
                        x_hat = m(x_hat)
                        vgg_xhat_features.append(x_hat)
                        x_hat.detach_()  # reset gradient from output of conv layer
                    else:
                        x_hat = m(x_hat)

        loss19 = 0
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            loss19 += l1_loss(x_feat, xhat_feat)

        loss = vgg19_weight * loss19 + vggface_weight * lossface

        return loss