""" Toolkit for ploting pose vectors
"""

# Imports
## mathematics
import numpy as np

## visualization
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.video.VideoClip import DataVideoClip

# CONSTANTS

class Colors:
    BLACK  = (0, 0, 0)
    RED    = (1, 0, 0)
    GREEN  = (0, 1, 0)
    BLUE   = (0, 0, 1)
    CYAN = (0, 1, 1)
    PURPLE = (1, 0, 1)

POSE_CONNECTIONS = [(15, 21), (16, 20), (18, 20), ( 3,  7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31), ( 6,  8), (15, 17), (24, 26), (16, 22), ( 4,  5), ( 5,  6), (29, 31), (12, 24), (23, 24), ( 0,  1), ( 9, 10), ( 1,  2), ( 0,  4), (11, 13), (30, 32), (28, 32), (15, 19), (16, 18), (25, 27), (26, 28), (12, 14), (17, 19), ( 2,  3), (11, 12), (27, 29), (13, 15)]
HAND_CONNECTIONS = [( 3,  4), ( 0,  5), (17, 18), ( 0, 17), (13, 14), (13, 17), (18, 19), ( 5,  6), ( 5,  9), (14, 15), ( 0,  1), ( 9, 10), ( 1,  2), ( 9, 13), (10, 11), (19, 20), ( 6,  7), (15, 16), ( 2,  3), (11, 12), ( 7,  8)]

N_POSE_LANDMARKS = max([index for connection in POSE_CONNECTIONS for index in connection]) + 1
N_HAND_LANDMARKS = max([index for connection in HAND_CONNECTIONS for index in connection]) + 1

ALL_CONNECTIONS  = sorted(POSE_CONNECTIONS) \
                 + [(N_POSE_LANDMARKS+start, N_POSE_LANDMARKS+end) 
                    for start, end in sorted(HAND_CONNECTIONS)] \
                 + [(N_POSE_LANDMARKS+N_HAND_LANDMARKS+start, N_POSE_LANDMARKS+N_HAND_LANDMARKS+end) 
                    for start, end in sorted(HAND_CONNECTIONS)]

N_ALL_LANDMARKS = max([index for connection in ALL_CONNECTIONS for index in connection]) + 1

def _get_default_connection_color(start=None, end=None):
    return Colors.BLACK

def _get_hand_connection_color(start, end): 
    return Colors.GREEN if end   <=  4 else \
           Colors.BLUE  if start >= 17 else _get_default_connection_color(start, end)

def _get_pose_connection_color(start, end): 
    return Colors.CYAN if (start % 2 == 1 and end % 2 == 1) else \
           Colors.PURPLE if (start % 2 == 0 and end % 2 == 0) else _get_default_connection_color(start, end)

def _get_all_connection_color(start, end):
    if start < N_POSE_LANDMARKS and end < N_POSE_LANDMARKS:
        color = _get_pose_connection_color(start, end)

    elif start < N_POSE_LANDMARKS+N_HAND_LANDMARKS and end < N_POSE_LANDMARKS+N_HAND_LANDMARKS:
        color = _get_hand_connection_color(start-N_POSE_LANDMARKS, end-N_POSE_LANDMARKS)

    elif start < N_POSE_LANDMARKS+N_HAND_LANDMARKS*2 and end < N_POSE_LANDMARKS+N_HAND_LANDMARKS*2:
        color = _get_hand_connection_color(start-N_POSE_LANDMARKS-N_HAND_LANDMARKS, end-N_POSE_LANDMARKS-N_HAND_LANDMARKS)

    else:
        color = _get_default_connection_color(start, end)

    return color

def infer_connections(n_landmarks):
    return  POSE_CONNECTIONS if n_landmarks == N_POSE_LANDMARKS else \
            HAND_CONNECTIONS if n_landmarks == N_HAND_LANDMARKS else \
            ALL_CONNECTIONS  if n_landmarks == N_ALL_LANDMARKS  else None

def _get_new_plt_fig(axis_lims=None, elev=15,azim=25, fig_width=10, fig_height=10):
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev = elev, azim = azim, vertical_axis='y')
    ax.set_box_aspect([1, 1, 1])

    if axis_lims:
        ax.set_xlim(axis_lims[0], axis_lims[1])
        ax.set_ylim(axis_lims[2], axis_lims[3])
        ax.set_zlim(axis_lims[4], axis_lims[5])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return fig, ax

# put landmarks on  3D graph
def plot_landmarks(landmarks, connections=None,
                    fig = None, ax = None, axis_lims = None, elev = 15, azim = 25,
                    landmarks_color = Colors.RED, landmark_size=5,
                ):
    """
    Plots a single pose & hands vector in 3D.
    Arguments:
        landmarks   - a numpy array of dimention (n,3) containing n landmarks (x, y, z).
        connections - tuples of indices of the landmarks to be connected by a line.
        axis_lims   - the upper and lower limits of the 3D plot axis [x_lower, x_upper, y_lower, y_upper, z_lower, z_upper]
    Returns:
        numpy array representing an image containing a single 3D plot.
    """

    # set up a matplotlib plot
    if fig is None or ax is None:
        fig, ax = _get_new_plt_fig(axis_lims=axis_lims, elev=elev, azim=azim)

    if connections is None:
        connections = infer_connections(len(landmarks))
    if connections is None:
        raise ValueError('connections cannot be None, provide a list of tuples of pairs of landmark indices')

    # plot landmarks as dots
    ax.scatter3D(xs=landmarks[:, 0], ys=landmarks[:, 1], zs=landmarks[:, 2],
                 color=landmarks_color, linewidth=landmark_size)

    get_connection_color =  _get_hand_connection_color if connections == HAND_CONNECTIONS else \
                            _get_pose_connection_color if connections == POSE_CONNECTIONS else \
                            _get_all_connection_color  if connections == ALL_CONNECTIONS  else \
                            _get_default_connection_color

    # draw lines to connect landmarks
    for connection in connections:
        start, end = connection

        color = get_connection_color(start,end)

        ax.plot3D(
            xs = [landmarks[start, 0], landmarks[end, 0]],
            ys = [landmarks[start, 1], landmarks[end, 1]],
            zs = [landmarks[start, 2], landmarks[end, 2]],
            color = color,
            linewidth = 2
        )

    return fig, ax

def _get_box_coord_ranges(multi_frame_landmarks):
    all_landmarks = np.stack(multi_frame_landmarks)

    x0, x1 = np.min(all_landmarks[:,:,0]), np.max(all_landmarks[:,:,0])
    y0, y1 = np.min(all_landmarks[:,:,1]), np.max(all_landmarks[:,:,1])
    z0, z1 = np.min(all_landmarks[:,:,2]), np.max(all_landmarks[:,:,2])

    return x0, x1, y0, y1, z0, z1

def _get_box(x0, x1, y0, y1, z0, z1):
    box_coords = np.array([
        [x0,y0,z0],
        [x0,y0,z1],
        [x0,y1,z0],
        [x0,y1,z1],
        [x1,y0,z0],
        [x1,y0,z1],
        [x1,y1,z0],
        [x1,y1,z1]
    ])

    box_connections = [
        (0,1), (0,2), (0,4),
        (7,6), (7,5), (7,3),
        (1,3), (3,2), (2,6),
        (6,4), (4,5), (5,1),
    ]

    return box_coords, box_connections

def plot_multi_frame_landmarks(multi_frame_landmarks, connections=None,
                                landmarks_color = Colors.RED,
                                axis_lims = None, elev = 15, azim = 25):

    x0, x1, y0, y1, z0, z1 = _get_box_coord_ranges(multi_frame_landmarks)
    box_coords, box_connections = _get_box(x0, x1, y0, y1, z0, z1)
    
    fig, ax = _get_new_plt_fig(axis_lims=axis_lims, elev=elev, azim=azim, fig_width=10*len(multi_frame_landmarks))
    
    for i, landmarks in enumerate(multi_frame_landmarks):
        x_shift = i * (x1 - x0)

        fig, ax = plot_landmarks(box_coords + x_shift, box_connections, fig = fig, ax = ax, axis_lims = axis_lims, elev = elev, azim = azim, landmarks_color = _get_default_connection_color(), landmark_size=3)
        fig, ax = plot_landmarks(landmarks  + x_shift,     connections, fig = fig, ax = ax, axis_lims = axis_lims, elev = elev, azim = azim, landmarks_color = landmarks_color)

    return fig, ax

def _fig_to_image(fig, ax=None):
    return mplfig_to_npimage(fig)

def landmarks_to_image(landmarks, connections=None,
                       landmarks_color = Colors.RED,
                       axis_lims = None, elev = 15, azim = 25):

    if connections is None:
        connections = infer_connections(len(landmarks))

    fig, _ = plot_landmarks(landmarks, connections, axis_lims = axis_lims, elev = elev, azim = azim,
                    landmarks_color = landmarks_color)

    image = _fig_to_image(fig)

    plt.clf()
    plt.close()

    return image

def landmarks_to_video(multi_frame_landmarks, connections=None,
                       landmarks_color = Colors.RED,
                       axis_lims = None, elev = 15, azim = 25,
                       fps=24, rotate=True):
                       
    if connections is None:
        connections = infer_connections(len(multi_frame_landmarks[0]))

    if axis_lims is None:
        scale = 0.1
        x0, x1, y0, y1, z0, z1 = _get_box_coord_ranges(multi_frame_landmarks)
        axis_lims = [
            x0 - abs(x0*scale), x1 + abs(x1*scale), 
            y0 - abs(y0*scale), y1 + abs(y1*scale), 
            z0 - abs(z0*scale), z1 + abs(z1*scale), 
        ]

    fig, ax = _get_new_plt_fig(axis_lims=axis_lims, elev=elev, azim=azim)

    fig, ax = plot_landmarks(multi_frame_landmarks[0], connections, fig = fig, ax = ax, axis_lims = axis_lims, elev = elev, azim = azim, landmarks_color = landmarks_color)

    n_frames = len(multi_frame_landmarks)

    def data_to_frame(t):
        
        ax[0].set_xdata(multi_frame_landmarks[t][:,0])
        ax[0].set_ydata(multi_frame_landmarks[t][:,1])
        ax[0].set_zdata(multi_frame_landmarks[t][:,2])
        
        if rotate:
            ax.view_init(elev = elev - elev*t*2/n_frames, azim = azim - azim*t*2/n_frames)

        return _fig_to_image(fig)

    timesteps = range(1, n_frames)
    clip = DataVideoClip(timesteps, data_to_frame, fps=fps)

    return clip