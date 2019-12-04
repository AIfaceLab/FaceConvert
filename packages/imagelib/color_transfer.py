import numpy as np
import cv2


def channel_hist_match(source, template, hist_match_threshold=255, mask=None):
    # Code borrowed from:
    # https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    masked_source = source
    masked_template = template

    if mask is not None:
        masked_source = source * mask
        masked_template = template * mask

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    masked_source = masked_source.ravel()
    masked_template = masked_template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    ms_values, mbin_idx, ms_counts = np.unique(source, return_inverse=True,
                                               return_counts=True)
    mt_values, mt_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles = hist_match_threshold * s_quantiles / s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles = 255 * t_quantiles / t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def color_hist_match(src_im, tar_im, hist_match_threshold=255):
    h, w, c = src_im.shape
    matched_R = channel_hist_match(
        src_im[:, :, 0], tar_im[:, :, 0], hist_match_threshold, None)
    matched_G = channel_hist_match(
        src_im[:, :, 1], tar_im[:, :, 1], hist_match_threshold, None)
    matched_B = channel_hist_match(
        src_im[:, :, 2], tar_im[:, :, 2], hist_match_threshold, None)

    to_stack = (matched_R, matched_G, matched_B)
    for i in range(3, c):
        to_stack += (src_im[:, :, i],)

    matched = np.stack(to_stack, axis=-1).astype(src_im.dtype)
    return matched
