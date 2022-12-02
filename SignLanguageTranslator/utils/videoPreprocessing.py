"""Extract landmarks from video
"""

import cv2
import numpy as np
import mediapipe as mp

BACKEND = "mediapipe"

# Mediapipe Solutions
hands = mp.solutions.hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
    max_num_hands=2,
)

pose = mp.solutions.pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=2
)


def mplandmark_to_nparray(landmarks, has_visibility):
    return np.array(
        [
            ([l.x, l.y, l.z, l.visibility] if has_visibility else [l.x, l.y, l.z])
            for l in landmarks.landmark
        ]
    )


def organize_mp_hand_landmarks_to_nparray(multi_hand_landmarks, multi_handedness):
    multi_hand_landmarks = np.zeros((42, 3))

    if multi_hand_landmarks:
        for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
            if handedness.classification[0].label == "Left":
                multi_hand_landmarks[:21] = mplandmark_to_nparray(
                    hand_landmarks, has_visibility=False
                )
            else:
                multi_hand_landmarks[21:] = mplandmark_to_nparray(
                    hand_landmarks, has_visibility=False
                )

    return multi_hand_landmarks


# process a video frame with mediapipe
def extract_pose_and_hands_landmarks_from_frame(frame, flipped=False):
    """
    Extracts the pose & hands landmarks from a frame of a video clip.
    Arguments:
        frame - A Numpy array of shape (None, None, 3) representing an image from a video clip containing a person.
    Returns:
        A Tuple of Numpy arrays of shapes (33, 4), (33, 4), (42, 3), (42, 3)
        where each row is a landmark (x, y, z, [visibility]).
        First 2 items in the tuple are pose vectors (image & world)
        followed by 2 hand vectors (image & world) each with first 21 landmarks for left hand and next 21 for right hand.
    """

    # Make Detections
    pose_results = pose.process(frame)
    hand_results = hands.process(frame)

    # Extract flattened Pose landmarks
    pose_landmarks = (
        mplandmark_to_nparray(pose_results.pose_landmarks, has_visibility=True)
        if pose_results.pose_landmarks
        else np.zeros((33, 4))
    )

    # Extract flattened Pose World landmarks
    pose_world_landmarks = (
        mplandmark_to_nparray(pose_results.pose_world_landmarks, has_visibility=True)
        if pose_results.pose_world_landmarks
        else np.zeros((33, 4))
    )

    multi_hand_landmarks = organize_mp_hand_landmarks_to_nparray(
        hand_results.multi_hand_landmarks, hand_results.multi_handedness
    )
    multi_hand_world_landmarks = organize_mp_hand_landmarks_to_nparray(
        hand_results.multi_hand_world_landmarks, hand_results.multi_handedness
    )

    return (
        pose_landmarks,
        pose_world_landmarks,
        multi_hand_landmarks,
        multi_hand_world_landmarks,
    )


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
    (
        pose_landmarks,
        pose_world_landmarks,
        multi_hand_landmarks,
        multi_hand_world_landmarks,
    ) = ([], [], [], [])

    cap = cv2.VideoCapture(videoPath)

    for frame_no in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        results = extract_pose_and_hands_landmarks_from_frame(
            cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        )
        pose_landmarks.append(results[0])
        pose_world_landmarks.append(results[1])
        multi_hand_landmarks.append(results[2])
        multi_hand_world_landmarks.append(results[3])

    cap.release()

    return (
        np.stack(pose_landmarks),
        np.stack(pose_world_landmarks),
        np.stack(multi_hand_landmarks),
        np.stack(multi_hand_world_landmarks),
    )
