import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def load_skeletons_image(path: str) -> np.ndarray:
    """
    Load skeletons image from a text file.

    :param path: (str) - The path to the skeletons image text file.
    :return: (np.ndarray) - A numpy array containing the loaded skeletons image.
    """
    return np.loadtxt(path)


def extract_joint_points(skeletons_image: np.ndarray, bones: np.ndarray) -> np.ndarray:
    """
    Extract joint points from skeletons image and create display data.

    :param skeletons_image: (np.ndarray) - Input skeletons image data.
    :param bones: (np.ndarray) - An array representing the bone connections.
    :return: (np.ndarray) - A numpy array containing extracted joint points for display.
    """
    num_images = skeletons_image.shape[0]
    num_joints = bones.shape[0]
    skeletons_display = np.zeros((num_images, 2, 2, num_joints))

    for id_image in range(num_images):
        ske = skeletons_image[id_image, :]
        x = np.zeros((2, num_joints))
        y = np.zeros((2, num_joints))

        for idx_bones in range(num_joints):
            joint1, joint2 = bones[idx_bones]
            pt1 = ske[joint1 * 2: joint1 * 2 + 2]
            pt2 = ske[joint2 * 2: joint2 * 2 + 2]
            x[0, idx_bones], x[1, idx_bones] = pt1[0], pt2[0]
            y[0, idx_bones], y[1, idx_bones] = pt1[1], pt2[1]

        skeletons_display[id_image, 0, :, :] = x
        skeletons_display[id_image, 1, :, :] = y

    return skeletons_display


def main() -> None:
    """
    Main function to visualize extracted joint points on depth images.
    """
    root_directory = '../data/SHREC/'
    idx_gesture, idx_subject, idx_finger, idx_essai = 1, 1, 1, 1

    bones = np.array([[0, 1], [0, 2], [2, 3], [3, 4], [4, 5], [1, 6], [6, 7], [7, 8],
                      [8, 9], [1, 10], [10, 11], [11, 12], [12, 13], [1, 14], [14, 15],
                      [15, 16], [16, 17], [1, 18], [18, 19], [19, 20], [20, 21]])

    path_gesture = os.path.join(root_directory,
                                f'gesture_{idx_gesture}/finger_{idx_finger}/subject_{idx_subject}/essai_{idx_essai}/')

    if os.path.isdir(path_gesture):
        path_skeletons_image = os.path.join(path_gesture, 'skeletons_image.txt')
        skeletons_image = load_skeletons_image(path_skeletons_image)
        pngDepthFiles = np.array([imageio.imread(os.path.join(path_gesture, f'{id_image}_depth.png'))
                                  for id_image in range(skeletons_image.shape[0])])
        skeletons_display = extract_joint_points(skeletons_image, bones)

        for id_image in range(skeletons_image.shape[0]):
            plt.clf()
            plt.imshow(pngDepthFiles[id_image, :])
            plt.plot(skeletons_display[id_image, 0, :, :], skeletons_display[id_image, 1, :, :],
                     linewidth=2.5)
            plt.pause(0.01)
    else:
        print(f'There is no gesture in the path {path_gesture}')


if __name__ == "__main__":
    main()
