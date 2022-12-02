"""load video/landmarks/text data from files
"""

# create a clip info table, fps resolution etc

# json/jsonl

import csv
import json
from os.path import dirname, exists, join

import moviepy.editor as mpy
import numpy as np


class SignDataFileAttributes:

    landmarks_types = [
        "Pose_Landmarks",
        "Pose_World_Landmarks",
        "Hand_Landmarks",
        "Hand_World_Landmarks",
    ]

    def __init__(
        self,
        word,
        file_type,
        person_no="101",
        camera_angle="front",
        signs_collection="HFAD_Book1",
        landmarks_type="",
        dataset_directory=None,
    ) -> None:

        self.word = word
        self.person_no = person_no
        self.camera_angle = camera_angle
        self.signs_collection = signs_collection
        self.file_path = None

        if dataset_directory is None:
            dataset_directory = join(
                dirname(dirname(__file__)),
                "datasets",
                "Signs_recordings",
            )
        if not exists(dataset_directory):
            raise ValueError(f"directory:'{dataset_directory}' to Signs dataset does not exit.")

        self.dataset_directory = dataset_directory

        self.video_or_preprocessed = (
            "Videos"
            if str(file_type).lower() in ["video", "videos", "mp4", ".mp4"]
            else "Landmarks"
            if str(file_type).lower() in ["landmarks", "preprocessed", "csv", ".mp4"]
            else None
        )

        if self.video_or_preprocessed is None:
            raise ValueError("unknown file_type of data")

        self.extension = "mp4" if self.video_or_preprocessed == "Videos" else "csv"

        if landmarks_type in [""] + SignDataFileAttributes.landmarks_types:
            self.landmarks_type = landmarks_type

        else:
            raise ValueError(
                f'Invalid landmarks_type. Use one from {[""]+SignDataFileAttributes.landmarks_types}'
            )

        if self.video_or_preprocessed == "Videos":
            assert landmarks_type == ""

    def set_file_path(self, file_path):
        self.file_path = file_path


def create_file_path(
    word,
    file_type,
    person_no="101",
    camera_angle="front",
    signs_collection="HFAD_Book1",
    landmarks_type="",
    dataset_directory=None,
):
    file_attrs = SignDataFileAttributes(
        word,
        file_type,
        person_no=person_no,
        camera_angle=camera_angle,
        signs_collection=signs_collection,
        landmarks_type=landmarks_type,
        dataset_directory=dataset_directory,
    )

    return join(
        file_attrs.dataset_directory,
        file_attrs.video_or_preprocessed,
        file_attrs.landmarks_type,
        file_attrs.signs_collection,
        f"person{file_attrs.person_no}",
        f"{file_attrs.word}_person{file_attrs.person_no}_{file_attrs.camera_angle}.{file_attrs.extension}",
    )


def load_video(
    word,
    person_no="101",
    camera_angle="front",
    signs_collection="HFAD_Book1",
    dataset_directory=None,
):
    file_path = create_file_path(
        word,
        "mp4",
        person_no=person_no,
        camera_angle=camera_angle,
        signs_collection=signs_collection,
        dataset_directory=dataset_directory,
    )
    return mpy.VideoFileClip(file_path)

def _read_landmarks_from_file(file_path):
    return np.loadtxt(file_path, skiprows=1, delimiter=",")


def load_landmarks(
    word,
    person_no="101",
    camera_angle="front",
    signs_collection="HFAD_Book1",
    landmarks_type="Pose_World_Landmarks",
    drop_visibility=False,
    dataset_directory=None,
):
    if landmarks_type in SignDataFileAttributes.landmarks_types:
        file_path = create_file_path(
            word,
            "csv",
            person_no=person_no,
            camera_angle=camera_angle,
            signs_collection=signs_collection,
            landmarks_type=landmarks_type,
            dataset_directory=dataset_directory,
        )
        return _read_landmarks_from_file(file_path)

    elif str(landmarks_type).lower() == "all":
        file_path = create_file_path(
            word,
            "csv",
            person_no=person_no,
            camera_angle=camera_angle,
            signs_collection=signs_collection,
            landmarks_type="pose",
        )
        pose = _read_landmarks_from_file(file_path)
        file_path = create_file_path(
            word,
            "csv",
            person_no=person_no,
            camera_angle=camera_angle,
            signs_collection=signs_collection,
            landmarks_type="hand",
        )
        hand = _read_landmarks_from_file(file_path)

        if drop_visibility:
            pose = pose.reshape((-1, 33, 4))[:, :, :3].reshape((-1, 3 * 33))

        return np.concatenate([pose, hand], axis=0)

    else:
        raise ValueError(
            'Invalid landmark type. Use one from ["all", "Pose_Landmarks", "Pose_World_Landmarks", "Hand_Landmarks", "Hand_World_Landmarks"]'
        )


POSE_HEADER = [h for i in range(0, 33) for h in [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]]
# POSE_HEADER = ['x0','y0','z0','v0',   'x1','y1','z1','v1',   ...  ,    'x32','y32','z32','v32']
HAND_HEADER = [h for i in range(0, 42) for h in [f"x{i}", f"y{i}", f"z{i}"]]
# HAND_HEADER = ['x0','y0','z0',        'x1','y1','z1',        ...  ,    'x41','y41','z41'      ]


def write_landmarks_in_csv(landmarks, path, header=None):
    """
    Saves landmarks of a video clip in a csv file.
    Arguments:
        landmarks - numpy array either of shape (None, 33, 4) for pose or (None, 42, 4) for hands
                    containing landmarks of each frame.
        path      - path where csv file should be created.
        header    - CSV header
    """
    if header is None:
        header = (
            POSE_HEADER
            if landmarks.shape[1] == len(POSE_HEADER)
            else HAND_HEADER
            if landmarks.shape[1] == len(HAND_HEADER)
            else None
        )

    with open(path, mode="w", newline="") as f:
        csv_writer = csv.writer(
            f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        if header:
            csv_writer.writerow(header)

        for frame in landmarks:
            csv_writer.writerow(frame.flatten())  # 1 frame per row


def write_landmarks_in_json(landmarks, landmarks_type, attrs: SignDataFileAttributes,  path):
    assert landmarks_type in SignDataFileAttributes.landmarks_types

    if exists(path):
        with open(path) as f:
            data = json.loads(f.read())
    else:
        data = dict()

    if landmarks_type not in data:
        data[landmarks_type] = dict()

    # for something in somethings:

    with open(path, 'w') as f:
        f.write(json.dumps(data))