import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast

from api import PRN

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture


def main(args):
    if args.isShow or args.isTexture:
        import cv2
        from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box

    # ---- init PRN
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # GPU number, -1 for CPU
    prn = PRN(is_dlib=args.isDlib)

    # ------------- load data
    image_folder = args.inputDir
    save_folder = args.outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    types = ("*.jpg", "*.png")
    image_path_list = []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)

    for i, image_path in enumerate(image_path_list):

        name = image_path.strip().split("/")[-1][:-4]

        # read image
        image = imread(image_path)
        [h, w, c] = image.shape
        if c > 3:
            image = image[:, :, :3]

        # the core: regress position map
        if args.isDlib:
            max_size = max(image.shape[0], image.shape[1])
            if max_size > 1000:
                image = rescale(image, 1000.0 / max_size)
                image = (image * 255).astype(np.uint8)
            pos = prn.process(image)  # use dlib to detect face
        else:
            if image.shape[0] == image.shape[1]:
                image = resize(image, (256, 256))
                pos = prn.net_forward(
                    image / 255.0
                )  # input image has been cropped to 256x256
            else:
                box = np.array(
                    [0, image.shape[1] - 1, 0, image.shape[0] - 1]
                )  # cropped with bounding box
                pos = prn.process(image, box)

        image = image / 255.0
        if pos is None:
            continue

        if args.is3d or args.isMat or args.isPose or args.isShow:
            # 3D vertices
            vertices = prn.get_vertices(pos)
            if args.isFront:
                save_vertices = frontalize(vertices)
            else:
                save_vertices = vertices.copy()
            save_vertices[:, 1] = h - 1 - save_vertices[:, 1]

        if args.isImage:
            imsave(os.path.join(save_folder, name + ".jpg"), image)

        if args.is3d:
            # corresponding colors
            colors = prn.get_colors(image, vertices)

            if args.isTexture:
                if args.texture_size != 256:
                    pos_interpolated = resize(
                        pos, (args.texture_size, args.texture_size), preserve_range=True
                    )
                else:
                    pos_interpolated = pos.copy()
                texture = cv2.remap(
                    image,
                    pos_interpolated[:, :, :2].astype(np.float32),
                    None,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0),
                )
                if args.isMask:
                    vertices_vis = get_visibility(vertices, prn.triangles, h, w)
                    uv_mask = get_uv_mask(
                        vertices_vis,
                        prn.triangles,
                        prn.uv_coords,
                        h,
                        w,
                        prn.resolution_op,
                    )
                    uv_mask = resize(
                        uv_mask,
                        (args.texture_size, args.texture_size),
                        preserve_range=True,
                    )
                    texture = texture * uv_mask[:, :, np.newaxis]
                write_obj_with_texture(
                    os.path.join(save_folder, name + ".obj"),
                    save_vertices,
                    prn.triangles,
                    texture,
                    prn.uv_coords / prn.resolution_op,
                )  # save 3d face with texture(can open with meshlab)
            else:
                write_obj_with_colors(
                    os.path.join(save_folder, name + ".obj"),
                    save_vertices,
                    prn.triangles,
                    colors,
                )  # save 3d face(can open with meshlab)

        if args.isDepth:
            depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
            depth = get_depth_image(vertices, prn.triangles, h, w)
            imsave(os.path.join(save_folder, name + "_depth.jpg"), depth_image)
            sio.savemat(
                os.path.join(save_folder, name + "_depth.mat"), {"depth": depth}
            )

        if args.isMat:
            sio.savemat(
                os.path.join(save_folder, name + "_mesh.mat"),
                {"vertices": vertices, "colors": colors, "triangles": prn.triangles},
            )

        if args.isKpt or args.isShow:
            # get landmarks
            kpt = prn.get_landmarks(pos)
            np.savetxt(os.path.join(save_folder, name + "_kpt.txt"), kpt)

        if args.isPose or args.isShow:
            # estimate pose
            camera_matrix, pose = estimate_pose(vertices)
            np.savetxt(os.path.join(save_folder, name + "_pose.txt"), pose)
            np.savetxt(
                os.path.join(save_folder, name + "_camera_matrix.txt"), camera_matrix
            )

            np.savetxt(os.path.join(save_folder, name + "_pose.txt"), pose)

        if args.isShow:
            # ---------- Plot
            image_pose = plot_pose_box(image, camera_matrix, kpt)
            cv2.imshow("sparse alignment", plot_kpt(image, kpt))
            cv2.imshow("dense alignment", plot_vertices(image, vertices))
            cv2.imshow("pose", plot_pose_box(image, camera_matrix, kpt))
            cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network"
    )

    parser.add_argument(
        "-i",
        "--inputDir",
        default="C:\PRNet\TestImages\TEST_results",
        type=str,
        help="path to the input directory, where input images are stored.",
    )
    parser.add_argument(
        "-o",
        "--outputDir",
        default="C:\PRNet\TestImages\TEST_results",
        type=str,
        help="path to the output directory, where results(obj,txt files) will be stored.",
    )
    parser.add_argument("--gpu", default="0", type=str, help="set gpu id, -1 for CPU")
    parser.add_argument(
        "--isDlib",
        default=True,
        type=ast.literal_eval,
        help="whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance",
    )
    parser.add_argument(
        "--is3d",
        default=True,
        type=ast.literal_eval,
        help="whether to output 3D face(.obj). default save colors.",
    )
    parser.add_argument(
        "--isMat",
        default=False,
        type=ast.literal_eval,
        help="whether to save vertices,color,triangles as mat for matlab showing",
    )
    parser.add_argument(
        "--isKpt",
        default=False,
        type=ast.literal_eval,
        help="whether to output key points(.txt)",
    )
    parser.add_argument(
        "--isPose",
        default=False,
        type=ast.literal_eval,
        help="whether to output estimated pose(.txt)",
    )
    parser.add_argument(
        "--isShow",
        default=False,
        type=ast.literal_eval,
        help="whether to show the results with opencv(need opencv)",
    )
    parser.add_argument(
        "--isImage",
        default=False,
        type=ast.literal_eval,
        help="whether to save input image",
    )
    # update in 2017/4/10
    parser.add_argument(
        "--isFront",
        default=False,
        type=ast.literal_eval,
        help="whether to frontalize vertices(mesh)",
    )
    # update in 2017/4/25
    parser.add_argument(
        "--isDepth",
        default=False,
        type=ast.literal_eval,
        help="whether to output depth image",
    )
    # update in 2017/4/27
    parser.add_argument(
        "--isTexture",
        default=False,
        type=ast.literal_eval,
        help="whether to save texture in obj file",
    )
    parser.add_argument(
        "--isMask",
        default=False,
        type=ast.literal_eval,
        help="whether to set invisible pixels(due to self-occlusion) in texture as 0",
    )
    # update in 2017/7/19
    parser.add_argument(
        "--texture_size",
        default=256,
        type=int,
        help="size of texture map, default is 256. need isTexture is True",
    )
    main(parser.parse_args())

if __name__ == "__main__":
    # Comment out or remove the argparse part if not needed
    # parser = argparse.ArgumentParser(...)
    # main(parser.parse_args())
    
    import gradio as gr
import numpy as np
from skimage.transform import resize
from api import PRN
from utils.write import write_obj_with_colors

# Function to handle the model inference
def generate_3d_face(image):
    try:
        # Initialize PRN without dlib for simplicity
        print("Initializing PRN model...")
        prn = PRN(is_dlib=False)  # Initialize the model without dlib

        print("Image shape before resizing:", np.array(image).shape)

        # Convert to numpy array and resize to ensure uniform input
        image = np.array(image)
        image = resize(image, (256, 256))  # Ensure the image is resized to 256x256

        print("Image shape after resizing:", image.shape)

        # Normalize the image and process it with the PRN model
        pos = prn.net_forward(image / 255.0)  # Process the image
        if pos is None:
            raise ValueError("Failed to process the image using PRN")

        print("3D position map processed.")

        # Get 3D vertices and generate the .obj file
        vertices = prn.get_vertices(pos)  # Get 3D vertices
        save_path = "output.obj"
        write_obj_with_colors(save_path, vertices, prn.triangles, image)  # Save .obj file

        print(f"3D face successfully saved at {save_path}")

        return save_path  # Return path to the saved .obj file

    except Exception as e:
        # Log the error and print it for debugging
        print(f"Error occurred: {e}")
        return f"Error: {e}"

# Gradio interface
iface = gr.Interface(
    fn=generate_3d_face,  # Function to run
    inputs="image",       # Input type is an image
    outputs="file",       # Output is a file (the .obj file)
    title="3D Face Reconstruction",
    description="Upload a 2D image, and the model will generate a 3D face in .obj format."
)

iface.launch()  # Launch the Gradio interface
