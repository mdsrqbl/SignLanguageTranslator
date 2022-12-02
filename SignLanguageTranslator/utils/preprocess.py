import cv2
import numpy as np
import mediapipe as mp

# Mediapipe Solutions
hands = mp.solutions.hands.Hands( min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1, max_num_hands=2)
pose  = mp.solutions.pose .Pose ( min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2)

# process a video frame with mediapipe
def extract_pose_and_hands_landmarks_from_frame(frame):
    """
    Extracts the pose & hands landmarks from a frame of a video clip.
    Arguments:
        frame - A Numpy array of shape (None, None, 3) representing an image from a video clip containing a person.
    Returns:
        A Tuple of Numpy arrays of shapes (33, 4), (33, 4), (42, 3), (42, 3)
        where each row is a landmark (x, y, z, [visibility]).
        First 2 items in the tuple are pose vectors (image & world)
        followed by 2 hand vectors (image & world) each with 21 landmarks for each hand.
    """

    # Make Detections
    presults = pose .process(         frame    )
    hresults = hands.process(cv2.flip(frame, 1))    # flipped for correct output for handedness (left or right)

    # initialize with zeros (used when landmarks are not detected)
    pose_landmarks             = np.zeros((33,4))
    pose_world_landmarks       = np.zeros((33,4))
    multi_hand_landmarks       = np.zeros((42,3))
    multi_hand_world_landmarks = np.zeros((42,3))

    # Extract flattened Pose landmarks
    if presults.pose_landmarks:
        pose_landmarks       = np.array([[l.x, l.y, l.z, l.visibility] for l in presults.pose_landmarks      .landmark])

    # Extract flattened Pose World landmarks
    if presults.pose_world_landmarks:
        pose_world_landmarks = np.array([[l.x, l.y, l.z, l.visibility] for l in presults.pose_world_landmarks.landmark])

    # Extract flattened Hand Landmarks
    if hresults.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(hresults.multi_hand_landmarks, hresults.multi_handedness):
            if(handedness.classification[0].label=="Left"):
                multi_hand_landmarks[:21] = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark])
            else:
                multi_hand_landmarks[21:] = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark])

    # Extract flattened Hand World Landmarks
    if hresults.multi_hand_world_landmarks:
        for hand_landmarks, handedness in zip(hresults.multi_hand_world_landmarks, hresults.multi_handedness):
            if(handedness.classification[0].label=="Left"):
                multi_hand_world_landmarks[:21] = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark])
            else:
                multi_hand_world_landmarks[21:] = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark])

    return pose_landmarks, pose_world_landmarks, multi_hand_landmarks, multi_hand_world_landmarks

##########################################################################################################################
##########################################################################################################################

def extract_pose_and_hands_landmarks_from_video(videoPath):
    """
    Extracts the pose landmarks from all frames of a video clip.
    Arguments:
        videoPath - Path of video file to be processed.
    Returns:
        Tuple of Numpy arrays of shapes (None, 33, 4), (None, 33, 4), (None, 42, 4), (None, 42, 4)
        where each row is a landmark (x, y, z, visibility).
        First 2 items in the tuple are pose vectors (image & world)
        followed by 2 hand vectors (image & world) each with 21 landmarks for each hand.
    """
    pose_landmarks, pose_world_landmarks, multi_hand_landmarks, multi_hand_world_landmarks = [],[],[],[]

    cap = cv2.VideoCapture(videoPath)

    for frame_no in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        results = extract_pose_and_hands_landmarks_from_frame(
                                cv2.cvtColor(   cap.read()[1]   , cv2.COLOR_BGR2RGB) )
        pose_landmarks             .append(  results[0]  )
        pose_world_landmarks       .append(  results[1]  )
        multi_hand_landmarks       .append(  results[2]  )
        multi_hand_world_landmarks .append(  results[3]  )

    cap.release()

    return (np.stack(  pose_landmarks              ),
            np.stack(  pose_world_landmarks        ),
            np.stack(  multi_hand_landmarks        ),
            np.stack(  multi_hand_world_landmarks  ))

##########################################################################################################################
##########################################################################################################################

import csv

pose_header = [title for i in range(0, 33)  for title in [f'x{i}', f'y{i}', f'z{i}', f'v{i}'] ]
#pose_header= ['x0','y0','z0','v0',   'x1','y1','z1','v1',   ...  ,    'x32','y32','z32','v32']
hand_header = [title for i in range(0, 42)  for title in [f'x{i}', f'y{i}', f'z{i}'         ] ]
#hand_header= ['x0','y0','z0',        'x1','y1','z1',        ...  ,    'x41','y41','z41'      ]

def write_landmarks_in_csv( landmarks, path ):
    """
    Saves landmarks of a video clip in a csv file.
    Arguments:
        landmarks - numpy array either of shape (None, 33, 4) for pose or (None, 42, 4) for hands 
                    containing landmarks of each frame.
        path      - path where csv file should be created.
    """

    header = pose_header if landmarks.shape[1] == 33 else hand_header

    with open(path, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow(header)
        for frame in landmarks:
            csv_writer.writerow(frame.flatten()) #1 frame per row

##########################################################################################################################
##########################################################################################################################

import os
import glob
from time import time
import datetime as dt
from IPython.display import clear_output

# Loop for all clips & write
def preprocess_this_folder(src_folder, dst_pose_folder, dst_pose_world_folder, dst_hand_folder, dst_hand_world_folder, overwrite=False):
    """
    Processes an entire folder of clips and generates 4 landmarks.csv files for each clip.
    Arguments:
        src_folder            - directory containing all the clips to be processed
        dst_pose_folder       - Directory where the generated pose (image) landmarks .csv files will be stored
        dst_pose_world_folder - Directory where the generated pose (world) landmarks .csv files will be stored
        dst_hand_folder       - Directory where the generated hand (image) landmarks .csv files will be stored
        dst_hand_world_folder - Directory where the generated hand (world) landmarks .csv files will be stored
        overwrite             - A boolean parameter to control whether existing .csv files in the destination folder should be overwritten (True) or not (False)
    """
    src_file_paths = glob.glob(os.path.join(src_folder,"*.mp4"))

    get_filename = lambda path: os.path.split(path)[-1].split('.')[0]

    if not overwrite:
        # get filenames that are already processed
        dst_pose_filenames       = [get_filename(path) for path in glob.glob( os.path.join(dst_pose_folder      ,"*.csv") )] 
        dst_pose_world_filenames = [get_filename(path) for path in glob.glob( os.path.join(dst_pose_world_folder,"*.csv") )] 
        dst_hand_filenames       = [get_filename(path) for path in glob.glob( os.path.join(dst_hand_folder      ,"*.csv") )] 
        dst_hand_world_filenames = [get_filename(path) for path in glob.glob( os.path.join(dst_hand_world_folder,"*.csv") )] 
        # keep only those video paths whose video files are not completely processed already
        src_file_paths = [srcPath for srcPath in src_file_paths
                          if get_filename(srcPath) not in
                          set(dst_pose_filenames).intersection(set(dst_pose_world_filenames), set(dst_hand_filenames), set(dst_hand_world_filenames))]

    start_time = time()
    for i, path in enumerate(src_file_paths):
        filename = os.path.split(path)[-1].split('.')[0]
        landmarks = extract_pose_and_hands_landmarks_from_video(path)
        write_landmarks_in_csv( landmarks[0], os.path.join(dst_pose_folder      , filename+'.csv'))
        write_landmarks_in_csv( landmarks[1], os.path.join(dst_pose_world_folder, filename+'.csv'))
        write_landmarks_in_csv( landmarks[2], os.path.join(dst_hand_folder      , filename+'.csv'))
        write_landmarks_in_csv( landmarks[3], os.path.join(dst_hand_world_folder, filename+'.csv'))

        clear_output(wait=True)
        now = time()
        print(f'Done:       {i+1}/{len(src_file_paths)} {filename}')
        print(f'Time Taken: {dt.timedelta( seconds=round(  now -start_time                                     ) )}')
        print(f'ETC:        {dt.timedelta( seconds=round( (now-start_time)*(len(src_file_paths)-(i+1))/(i+1)  ) )}')
        print(f'Rate:       {(i+1)/(now-start_time)*60:.3f} files per minute')

##########################################################################################################################
##########################################################################################################################