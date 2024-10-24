import sys
sys.path.append(r'C:\PRNet')  # Replace with the actual path to PRNet

import dlib
import cv2
import numpy as np
import os
from scipy.io import savemat
from api import PRN

# Define the facial landmark detector path
detector_path = (
    r"C:/deep3d/Deep3DFaceRecon_pytorch/shape_predictor_68_face_landmarks.dat"
)

# Load the detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(detector_path)

# Define the path to the image folder
image_folder = r"C:\PRNet\TestImages\Desha"  # Replace with your actual folder path

# Define the detections path (where to save .mat files)
detections_path = r"C:\PRNet\TestImages\Desha\detections"  # Replace if needed

# Ensure detections path exists
os.makedirs(detections_path, exist_ok=True)

# Function to generate dummy parameters similar to those in the provided .mat file
def generate_dummy_parameters():
    Illum_Para = np.random.rand(1, 10)
    Color_Para = np.random.rand(1, 7)
    Tex_Para = np.random.randn(199, 1) * 1000
    Shape_Para = np.random.randn(199, 1) * 1000
    Exp_Para = np.random.randn(29, 1)
    Pose_Para = np.random.randn(1, 7)
    roi = np.array([[162, 52, 1182, 1072]])
    return Illum_Para, Color_Para, Tex_Para, Shape_Para, Exp_Para, Pose_Para, roi

# Function to extract and save landmarks for an image in a .mat file
def process_image_to_mat(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            first_face = faces[0]
            landmarks = predictor(gray, first_face)

            pt2d = np.zeros((2, 21))
            for i in range(21):
                pt2d[0, i] = landmarks.part(i).x
                pt2d[1, i] = landmarks.part(i).y

            pt3d_68 = np.zeros((3, 68))
            for i in range(68):
                pt3d_68[0, i] = landmarks.part(i).x
                pt3d_68[1, i] = landmarks.part(i).y
                pt3d_68[2, i] = 0  # Dummy z-coordinate

            Illum_Para, Color_Para, Tex_Para, Shape_Para, Exp_Para, Pose_Para, roi = generate_dummy_parameters()

            filename, _ = os.path.splitext(os.path.basename(image_path))
            mat_filepath = os.path.join(detections_path, filename + ".mat")

            savemat(mat_filepath, {
                'pt2d': pt2d,
                'Illum_Para': Illum_Para,
                'Color_Para': Color_Para,
                'Tex_Para': Tex_Para,
                'Shape_Para': Shape_Para,
                'Exp_Para': Exp_Para,
                'Pose_Para': Pose_Para,
                'roi': roi,
                'pt3d_68': pt3d_68
            })

            print(f"Landmarks saved to: {mat_filepath}")
        else:
            print(f"No faces detected in: {image_path}")

# Loop through all images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        process_image_to_mat(image_path)
