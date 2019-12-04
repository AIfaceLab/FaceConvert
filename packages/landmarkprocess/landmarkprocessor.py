import numpy as np
import math
from enum import IntEnum
from ..mathlib.umeyama import umeyama
import cv2
# facetype-------------------------------------------------------------------------


class FaceType(IntEnum):
    HALF = 0,
    FULL = 1,
    HEAD = 2,
    AVATAR = 3,  # centered nose only
    MARK_ONLY = 4,  # no align at all, just embedded faceinfo
    QTY = 5

    @staticmethod
    def fromString(s):
        r = from_string_dict.get(s.lower())
        if r is None:
            raise Exception('FaceType.fromString value error')
        return r

    @staticmethod
    def toString(face_type):
        return to_string_list[face_type]


from_string_dict = {'half_face': FaceType.HALF,
                    'full_face': FaceType.FULL,
                    'head': FaceType.HEAD,
                    'avatar': FaceType.AVATAR,
                    'mark_only': FaceType.MARK_ONLY,
                    }
to_string_list = ['half_face',
                  'full_face',
                  'head',
                  'avatar',
                  'mark_only'
                  ]
# ----------------------------------------------------------------------------


mean_face_x = np.array([
    0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
    0.553364, 0.490127, 0.42689])

mean_face_y = np.array([
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
    0.784792, 0.824182, 0.831803, 0.824182])

landmarks_2D = np.stack([mean_face_x, mean_face_y], axis=1)


def get_transform_mat(image_landmarks, output_size, face_type, scale=1.0):
    if not isinstance(image_landmarks, np.ndarray):
        image_landmarks = np.array(image_landmarks)

    if face_type == FaceType.AVATAR:
        centroid = np.mean(image_landmarks, axis=0)

        mat = umeyama(image_landmarks[17:], landmarks_2D, True)[0:2]
        a, c = mat[0, 0], mat[1, 0]
        scale = math.sqrt((a * a) + (c * c))

        padding = (output_size / 64) * 32

        mat = np.eye(2, 3)
        mat[0, 2] = -centroid[0]
        mat[1, 2] = -centroid[1]
        mat = mat * scale * (output_size / 3)
        mat[:, 2] += output_size / 2
    else:
        if face_type == FaceType.HALF:
            padding = 0
        elif face_type == FaceType.FULL:
            padding = (output_size / 64) * 12
        elif face_type == FaceType.HEAD:
            padding = (output_size / 64) * 24
        else:
            raise ValueError('wrong face_type: ', face_type)

        mat = umeyama(image_landmarks[17:], landmarks_2D, True)[0:2]
        mat = mat * (output_size - 2 * padding)
        mat[:, 2] += padding
        mat *= (1 / scale)
        mat[:, 2] += -output_size*(((1 / scale) - 1.0) / 2)

    return mat


# get the mask of the image
def get_image_hull_mask(image_shape, image_landmarks, ie_polys=None):
    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int)

    hull_mask = np.zeros(image_shape[0:2]+(1,), dtype=np.float32)

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[0:9],
                        int_lmrks[17:18]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[8:17],
                        int_lmrks[26:27]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:20],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[24:27],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[19:25],
                        int_lmrks[8:9],
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:22],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[22:27],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    # nose
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(int_lmrks[27:36]), (1,))

    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)

    return hull_mask
