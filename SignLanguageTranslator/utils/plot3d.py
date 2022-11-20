import pandas as pd
import imutils
import moviepy.editor as mpy
import cv2
import numpy as np
from matplotlib import pyplot as plt

from preprocess import *

Pose_Connections = {(15, 21), (16, 20), (18, 20), (3,  7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31), (6,  8), (15, 17), (24, 26), (16, 22), (4,  5), (5,  6), (29, 31), (12, 24),
                    (23, 24), (0,  1), (9, 10), (1,  2), (0,  4), (11, 13), (30, 32), (28, 32), (15, 19), (16, 18), (25, 27), (26, 28), (12, 14), (17, 19), (2,  3), (11, 12), (27, 29), (13, 15)}
Hand_Connections = {(3,  4), (0,  5), (17, 18), (0, 17), (13, 14), (13, 17), (18, 19), (5,  6), (5,  9), (14, 15),
                    (0,  1), (9, 10), (1,  2), (9, 13), (10, 11), (19, 20), (6,  7), (15, 16), (2,  3), (11, 12), (7,  8)}

# make a single 3D plot
def plot_landmarks(landmarks, connections, axis_lims=None, elev=10,azim=10):
    """
    Plots a single pose & hands vector in 3D.
    Arguments:
        landmarks   - a numpy array of dimention (n,4) containing n landmarks (x,y,z,visibility).
        connections - tuples of indices of the landmarks to be connected by a line.
        axis_lims   - the upper and lower limits of the 3D plot axis [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper]
    Returns:
        numpy array of shape (400, 500, 3) representing an image containing a single 3D plot.
    """

    if np.sum(landmarks) == 0:
        return cv2.imread('empty3Dplot.png')

    # set up a matplotlib plot
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])

    if axis_lims != None:
        ax.set_xlim(axis_lims[0], axis_lims[1])
        ax.set_ylim(axis_lims[2], axis_lims[3])
        ax.set_zlim(axis_lims[4], axis_lims[5])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # plot landmarks as dots
    ax.scatter3D(xs=-landmarks[:, 2], ys=landmarks[:, 0],
                 zs=-landmarks[:, 1], color=(1, 0, 0), linewidth=5)

    # draw lines to connect landmarks
    for connection in connections:
        start, end = connection

        if connections == Hand_Connections:
            color = (0, 1, 0) if end   <=  4 else \
                    (0, 0, 1) if start >= 17 else (0, 0, 0)
        else:
            color = (0, 1, 0) if (start % 2 == 1 and end % 2 == 1) else \
                    (0, 0, 1) if (start % 2 == 0 and end % 2 == 0) else (0, 0, 0)

        ax.plot3D(
            xs=[-landmarks[start, 2], -landmarks[end, 2]],
            ys=[landmarks[start, 0],  landmarks[end, 0]],
            zs=[-landmarks[start, 1], -landmarks[end, 1]],
            color=color,
            linewidth=2)

    # draw the plot and extract it as an image (numpy array (height, width, channels))
    fig.canvas.draw()
    plt_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plt_img = plt_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return plt_img[170:570, 110:610]  # crop the white borders and return image

##########################################################################################################################
##########################################################################################################################


# plot landmarks into a video


def draw_video_plot(landmarks, fps, standing=True):
    """
    Plots a sequence of pose and hands vectors in a 3D video
    Arguments:
        landmarks - Numpy array of shape (None, 75, 4) containing a matrix from each frame. In that matrix, First 33 landmarks (x, y, z, visibility) are pose followed by left and right hand with 21 landmarks each.
        fps       - Frames rate of resulting video
        standing  - Boolean variable used to set axis limits if the person is standing or seated
    Returns:
        moviepy video clip object
    """

    # read plot labels
    pose_text, hand_text = cv2.imread(
        'pose_text.png'), cv2.imread('hand_text.png')

    # set plot axis limits
    # [-0.5,0.5, -0.5,0.5, -1.0,1.0] if standing else [-0.5,0.5]*3
    axis_limits = [-0.85, 0.65]*3

    plot_frames = list()
    for frame in landmarks:
        pose_plot = plot_landmarks(
            frame[:33], Pose_Connections, axis_lims=axis_limits)
        lhand_plot = plot_landmarks(
            frame[33:54], Hand_Connections, axis_lims=[-0.1, 0.1, 0.1, -0.1, -0.1, 0.1])
        rhand_plot = plot_landmarks(
            frame[54:], Hand_Connections, axis_lims=[-0.1, 0.1, 0.1, -0.1, -0.1, 0.1])

        hand_img = imutils.resize(cv2.hconcat(
            [rhand_plot, lhand_plot]), width=pose_plot.shape[1])

        plot_frames.append(cv2.vconcat(
            [pose_text, pose_plot, hand_text, hand_img]))

    return mpy.ImageSequenceClip(plot_frames, fps)

##########################################################################################################################
##########################################################################################################################


def read_landmarks_csv(path, step=1):
    return np.array(pd.read_csv(path)).reshape((-1, 75, 4))[::step]

##########################################################################################################################
##########################################################################################################################


def read_file_and_make_video_plot(path, from_video, step=1, standing=True):
    """
    Either opens a video to first extract pose and hand vectors 
    or opens a precomputed pose and hand vectors 
    and plots these in a 3D video
    Arguments:
        path       - Path to the file to be plotted
        from_video - Boolean variable telling whether the file is a video file or otherwise csv
        step       - Step size used to skip frames for faster output
        standing   - Boolean variable used to set axis limits if the person is standing or seated
    Returns:
        moviepy video clip object
    """

    if from_video:
        landmarks = extract_pose_and_hands_vector_from_video(path, step)

        # get video fps
        cap = cv2.videoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        video = draw_video_plot(landmarks, fps,     standing=standing)
    else:  # .csv file
        landmarks = read_landmarks_csv(path, step)

        video = draw_video_plot(landmarks, 30/step, standing=standing)

    return video

##########################################################################################################################
##########################################################################################################################


def write_video(vid, path, message=False):
    if message:
        print(f'Writing file:  {path} ,  {vid.duration:.3} sec')
    vid.write_videofile(path, audio=False, threads=8,
                        verbose=False, logger=None)

##########################################################################################################################
##########################################################################################################################
