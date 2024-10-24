import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from time import time

from api import PRN
from utils.write import write_obj_with_colors, write_obj_with_texture

# ---- init PRN
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU number, -1 for CPU
prn = PRN(is_dlib=True)

import cv2


# ------------- load data
image_folder = "TestImages/TEST/"
save_folder = "TestImages/TEST_results"
if not os.path.exists(save_folder):
    os.mkdirs(save_folder)

types = ("*.jpg", "*.png")
image_path_list = []
for files in types:
    image_path_list.extend(glob(os.path.join(image_folder, files)))
total_num = len(image_path_list)

for i, image_path in enumerate(image_path_list):
    # read image
    image = imread(image_path)

    # the core: regress position map
    if "AFLW2000" in image_path:
        mat_path = image_path.replace("jpg", "mat")
        info = sio.loadmat(mat_path)
        kpt = info["pt3d_68"]
        pos = prn.process(
            image, kpt
        )  # kpt information is only used for detecting face and cropping image
    else:
        pos = prn.process(image)  # use dlib to detect face

    image = image / 255.0

    # -- Basic Applications
    # get landmarks
    kpt = prn.get_landmarks(pos)
    # 3D vertices
    vertices = prn.get_vertices(pos)
    # corresponding colors
    colors = prn.get_colors(image, vertices)

    pos_interpolated = pos.copy()
    texture = cv2.remap(
        image,
        pos_interpolated[:, :, :2].astype(np.float32),
        None,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0),
    )

    # -- save
    name = image_path.strip().split("/")[-1][:-4]
    np.savetxt(os.path.join(save_folder, name + ".txt"), kpt)
    write_obj_with_colors(
        os.path.join(save_folder, name + ".obj"), vertices, prn.triangles, colors
    )  # save 3d face(can open with meshlab)
    # write_obj_with_texture(os.path.join(save_folder, name + '.obj'), vertices, prn.triangles, texture, prn.uv_coords/prn.resolution_op)#save 3d face with texture(can open with meshlab)
    sio.savemat(
        os.path.join(save_folder, name + "_mesh.mat"),
        {"vertices": vertices, "colors": colors, "triangles": prn.triangles},
    )
