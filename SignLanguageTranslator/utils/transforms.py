"""transform landmarks
"""

from typing import List, Tuple, Union

import numpy as np
import torch
from scipy.interpolate import interp1d as interpolate
from scipy.spatial.transform import Rotation as R

from .landmarksInfo import LandmarksInfo


def _arg_type_normalizer(*args) -> Union[List[float], Tuple[float, float]]:
    """turn arguments into (float,) or ((float,float),)

    Returns:
        list: list of floats or list of tuples of 2 floats
    """

    scalars = []
    sequences = []

    return_sequence = False

    for arg in args:
        if hasattr(arg, "__len__") and len(arg) > 1:
            return_sequence = True
            sequences.append(list(arg))
        else:
            scalars.append(arg)
            sequences.append([arg, arg])

    return sequences if return_sequence else scalars


def zoom_landmarks(
    landmarks: Union[np.ndarray, torch.Tensor],
    x_factor: Union[float, Tuple[float, float]],
    y_factor: Union[float, Tuple[float, float]],
    z_factor: Union[float, Tuple[float, float]],
) -> Union[np.ndarray, torch.Tensor]:
    """scale landmarks around their origin, separatly in each dimention

    Args:
        landmarks (Union[np.ndarray, torch.Tensor]): a 2D or 3D array of landmarks to be zoomed
        x_factor (Union[float, Tuple[float, float]]): scaling factor along x axis
        y_factor (Union[float, Tuple[float, float]]): scaling factor along y axis
        z_factor (Union[float, Tuple[float, float]]): scaling factor along z axis

    Raises:
        NotImplementedError: landmarks type other than numpy.ndarray or torch.Tensor are not handled.

    Returns:
        Union[np.ndarray, torch.Tensor]: scaled landmarks
    """
    # *np.random.normal(1,0.05,(3,2)).tolist() # for dynamic
    # *(np.random.rand(3)+0.5).tolist() # for static. right?

    x_factor, y_factor, z_factor = _arg_type_normalizer(x_factor, y_factor, z_factor)

    if not isinstance(x_factor, list):
        factor = [x_factor, y_factor, z_factor]
    else:
        n_frames = len(landmarks)
        factor = np.linspace(
            [x_factor[0], y_factor[0], z_factor[0]],
            [x_factor[1], y_factor[1], z_factor[1]],
            n_frames,
        )[:, np.newaxis, :]

    if isinstance(landmarks, torch.Tensor):
        factor = torch.Tensor(factor)

    elif isinstance(landmarks, np.ndarray):
        factor = np.array(factor)

    else:
        raise NotImplementedError("use numpy.ndarray or torch.Tensor")

    reshaper = LandmarksInfo.Reshaper()
    landmarks = reshaper.fold(landmarks)
    landmarks = landmarks * factor
    landmarks = reshaper.unfold(landmarks)

    return landmarks


def rotate_landmarks(
    landmarks: Union[np.ndarray, torch.Tensor],
    x_angle: Union[float, Tuple[float, float]],
    y_angle: Union[float, Tuple[float, float]],
    z_angle: Union[float, Tuple[float, float]],
    in_degrees: bool,
) -> Union[np.ndarray, torch.Tensor]:
    """Rotate landmarks around their origin, with a separate angle for every dimention.

    Args:
        landmarks (Union[np.ndarray, torch.Tensor]): a 2D or 3D array of landmarks to be rotated
        x_angle (Union[float, Tuple[float, float]]): angle of rotation around x axis
        y_angle (Union[float, Tuple[float, float]]): angle of rotation around y axis
        z_angle (Union[float, Tuple[float, float]]): angle of rotation around z axis
        in_degrees (bool): True if angles are in degrees, False for radians.

    Raises:
        NotImplementedError: landmarks type other than numpy.ndarray or torch.Tensor are not handled.

    Returns:
        Union[np.ndarray, torch.Tensor]: rotated landmarks
    """
    # *np.random.normal(0,60,(3,2)).tolist(), in_degrees=True

    x_angle, y_angle, z_angle = _arg_type_normalizer(x_angle, y_angle, z_angle)

    if not isinstance(x_angle, list):
        rotation_matrix = R.from_euler(
            "xyz", [x_angle, y_angle, z_angle], degrees=in_degrees
        ).as_matrix()

    else:
        n_frames = len(landmarks)
        rotation_matrix = np.stack(
            [
                R.from_euler("xyz", [x, y, z], degrees=in_degrees).as_matrix()
                for x, y, z in np.linspace(
                    [x_angle[0], y_angle[0], z_angle[0]],
                    [x_angle[1], y_angle[1], z_angle[1]],
                    n_frames,
                )
            ]
        )

    if isinstance(landmarks, np.ndarray):
        matmul = np.matmul

    elif isinstance(landmarks, torch.Tensor):
        matmul = torch.matmul
        rotation_matrix = torch.from_numpy(rotation_matrix)

    else:
        raise NotImplementedError("use numpy.ndarray or torch.Tensor")

    reshaper = LandmarksInfo.Reshaper()
    landmarks = reshaper.fold(landmarks)
    landmarks = matmul(landmarks, rotation_matrix)
    landmarks = reshaper.unfold(landmarks)

    return landmarks


def noisy_landmarks(
    landmarks: Union[np.ndarray, torch.Tensor],
    vary_through_time: bool,
    nlandmarks_mean_std: List[Tuple[int, float, float]] = [
        [11, 0.003, 0.0005],  # face
        [4, 0.006, 0.0025],  # shoulders & elbows
        [2, 0.004, 0.0008],  # pose wrists
        [6, 0.0025, 0.0005],  # pose hands
        [10, 0.016, 0.0050],  # hips & legs
        [42, 0.002, 0.0004],  # hands
    ],  # [[33, 0.004, 0.0005], [42, 0.001, 0.0002]],
    visibility_noise_scale_mean: float = 0.025,
    visibility_noise_scale_std: float = 0.010,
) -> Union[np.ndarray, torch.Tensor]:
    """Add gaussian noise of different scales to different landmarks.

    Args:
        landmarks (Union[np.ndarray, torch.Tensor]): a 2D or 3D array of landmarks
        vary_through_time (bool): True means the landmarks would get different noise value in diffent timesteps. False means same noise values added through all time steps.
        nlandmarks_mean_std (List[Tuple[int, float, float]], optional): each row tells how many landmarks should get noise of a scale determined by another gaussian distribution of that given mean and std. Defaults to [ [11, 0.003, 0.0005],  # face [4, 0.006, 0.0025],  # shoulders & elbows [2, 0.004, 0.0008],  # pose wrists [6, 0.0025, 0.0005],  # pose hands [10, 0.016, 0.0050],  # hips & legs [42, 0.002, 0.0004],  # hands ].
        visibility_noise_scale_mean (float, optional): the mean used by a gaussian distribution to determine the scale of noise added to the visibility feature. Defaults to 0.025.
        visibility_noise_scale_std (float, optional): the standard deviation used by a gaussian distribution to determine the scale of noise added to the visibility feature. Defaults to 0.010.

    Raises:
        NotImplementedError: landmarks type other than numpy.ndarray or torch.Tensor are not handled.

    Returns:
        Union[np.ndarray, torch.Tensor]: landmarks with noise added (which can be constant through time or changing)
    """

    if isinstance(landmarks, np.ndarray):
        normal = np.random.normal
        concatenate = np.concatenate

    elif isinstance(landmarks, torch.Tensor):
        normal = torch.normal
        concatenate = torch.concatenate

    else:
        raise NotImplementedError("use numpy.ndarray or torch.Tensor")

    no_detections_mask = landmarks == 0
    reshaper = LandmarksInfo.Reshaper()
    landmarks = reshaper.fold(landmarks)
    landmark_dim = -2
    time_dim = -3

    # determine noise shape
    noise_size = list(landmarks.shape)

    if not vary_through_time and landmarks.ndim > 2:
        noise_size[time_dim] = 1

    # generate noise
    noise = []
    for nlandmarks, mean, std in nlandmarks_mean_std:
        noise_size[landmark_dim] = nlandmarks
        noise_scale = abs(normal(mean, std, (1,)))
        noise.append(normal(0, noise_scale, noise_size))

    noise = concatenate(noise, axis=landmark_dim)

    assert (
        noise.shape[landmark_dim] == landmarks.shape[landmark_dim]
    ), "fix nlandmarks_mean_std to cover all landmarks"

    if reshaper.landmarks_visibility_backup is not None:
        visibility_noise_scale = abs(
            normal(visibility_noise_scale_mean, visibility_noise_scale_std, (1,))
        )
        visibility_noise = normal(
            0, visibility_noise_scale, reshaper.landmarks_visibility_backup.shape
        )
        reshaper.landmarks_visibility_backup = (
            reshaper.landmarks_visibility_backup + visibility_noise
        )
        reshaper.landmarks_visibility_backup[
            reshaper.landmarks_visibility_backup < 0
        ] = 0
        reshaper.landmarks_visibility_backup[
            reshaper.landmarks_visibility_backup > 1
        ] = 1

    landmarks = landmarks + noise
    landmarks = reshaper.unfold(landmarks)
    landmarks[no_detections_mask] = 0

    return landmarks


def _prepare_timesteps(
    n_steps: int,
    lower_limit: float = 0,
    upper_limit: float = 1,
    return_type=np.ndarray,
    add_random_steps: bool = False,
    n_random_steps_frac: float = None,
    min_n_linear_steps: int = 20,
    min_n_linear_steps_frac: float = 0.25,
    add_slow_spots: bool = False,
    n_slow_spots: int = None,
    n_slow_spots_frac: float = 0.02,
    slow_spots_steps_frac: float = 0.2,
) -> Union[np.ndarray, torch.Tensor]:
    """A mess of a function. Generates numbers in a given range which are evenly or randomly distributed or concentrated around some point(s).

    Args:
        n_steps (int): how many values to generate in the given range.
        lower_limit (float, optional): the lower bound of the range. Defaults to 0.
        upper_limit (float, optional): the upper bound of the range. Defaults to 1.
        return_type (type, optional): the type of object to return. Only np.ndarray and torch.Tensor are supported. Defaults to np.ndarray.
        add_random_steps (bool, optional): True means sample some values from uniform distribution. Defaults to False.
        n_random_steps_frac (float, optional): what percentage of total steps should be sampled from uniform distribution. If None, the ratio is randomly selected. Defaults to None.
        min_n_linear_steps (int, optional): minimum number of steps that should be evenly spaced and span the entire range. Defaults to 20.
        min_n_linear_steps_frac (float, optional): percentage of total steps at least which should be evenly spaced and span the entire range. Its an alternate to the above and maximum of both is used. Defaults to 0.25.
        add_slow_spots (bool, optional): True means sample some values from a normal distribution concentrated around some point. Defaults to False.
        n_slow_spots (int, optional): number of such concentrated points to use. If None, its selected randomly. Defaults to None.
        n_slow_spots_frac (float, optional): ratio of number of spots to total steps. Defaults to 0.02.
        slow_spots_steps_frac (float, optional): percentage of total steps to be used to make such spots. Defaults to 0.2.

    Raises:
        Exception: for some reason, no kind of values were generated.
        NotImplementedError: landmarks type other than numpy.ndarray or torch.Tensor are not handled.

    Returns:
        np.ndarray | torch.Tensor: an array of n sorted values between the given range.
    """

    steps = []
    remaining_steps = n_steps
    min_n_linear_steps = max(min_n_linear_steps, min_n_linear_steps_frac * n_steps)
    min_n_linear_steps = min(min_n_linear_steps, n_steps)

    if add_slow_spots:
        if n_slow_spots is None:
            n_slow_spots = round(
                n_steps * n_slow_spots_frac + np.random.rand(1).item() * 0.8 - 0.4
            )

        if n_slow_spots > 0:
            n_slow_spot_steps = slow_spots_steps_frac * n_steps / n_slow_spots
            n_slow_spots_steps = (
                np.random.rand(n_slow_spots) + 0.5  # / 2 +0.75
            ) * n_slow_spot_steps

            slow_spot_location = (
                np.random.rand(n_slow_spots) * (upper_limit - lower_limit) + lower_limit
            )

            slow_spot_scale = n_slow_spots_steps / n_steps / 4

            for loc, scale, n_samples in zip(
                slow_spot_location, slow_spot_scale, n_slow_spots_steps
            ):
                x = np.random.normal(loc, scale, int(n_samples))
                x = x[(x >= lower_limit) & (x <= upper_limit)]
                if len(x) > remaining_steps - min_n_linear_steps:
                    break
                steps.append(x)
                remaining_steps -= len(x)

    if add_random_steps:
        if n_random_steps_frac is None:
            n_random_steps_frac = np.random.rand(1)

        n_linear_steps = np.rint(
            max(min_n_linear_steps, remaining_steps * (1 - n_random_steps_frac))
        )
        n_random_steps = int(remaining_steps - n_linear_steps)

        steps.append(np.random.rand(n_random_steps))
        remaining_steps -= n_random_steps

    steps.append(np.linspace(lower_limit, upper_limit, remaining_steps))

    if len(steps) == 1:
        steps = steps[0]
    elif len(steps) > 1:
        steps = np.concatenate(steps)
    else:
        raise Exception("empty timesteps list")

    assert len(steps) == n_steps

    if return_type == np.ndarray:
        steps = np.sort(steps)
    elif return_type == torch.Tensor:
        steps = torch.Tensor(np.sort(steps))
    else:
        raise NotImplementedError("use np.ndarray or torch.Tensor as return_type")

    return steps


def _interpolate_multi_frame_landmarks(
    multi_frame_landmarks: Union[np.ndarray, torch.Tensor],
    old_timesteps: Union[np.ndarray, torch.Tensor],
    new_timesteps: Union[np.ndarray, torch.Tensor],
    time_dimention: int = 0,
    bad_detection_threshold: float = 0.1,
) -> Union[np.ndarray, torch.Tensor]:
    """Interpolate new points within the given data across a given dimention. Should create new frames across time dimention at certain timestamps.

    Args:
        multi_frame_landmarks (Union[np.ndarray, torch.Tensor]): a 3D array of landmarks to be interpolated
        old_timesteps (Union[np.ndarray, torch.Tensor]): independent (time) axis values that correspond to each subarray in data.
        new_timesteps (Union[np.ndarray, torch.Tensor]): independent (time) axis values for which subarrays should be created.
        time_dimention (int, optional): the dimention number of the landmarks array across which interpolation should take place. Defaults to 0.
        bad_detection_threshold (float, optional): Used to convert interpolated mask into anther mask. values in interpolated mask greater than this are converted to True and corresponding landmarks put equal to 0. Defaults to 0.1.

    Returns:
        Union[np.ndarray, torch.Tensor]: Interpolated Landmarks
    """

    assert len(old_timesteps) == multi_frame_landmarks.shape[time_dimention]

    no_detection_mask = multi_frame_landmarks == 0
    if no_detection_mask.shape[-1] in [
        LandmarksInfo.N_ALL_FEATURES,
        LandmarksInfo.N_POSE_FEATURES,
    ]:
        no_detection_mask[..., LandmarksInfo.VISIBILITY_INDEXES] = True

    f_landmarks = interpolate(old_timesteps, multi_frame_landmarks, axis=time_dimention)
    f_mask = interpolate(old_timesteps, no_detection_mask, axis=time_dimention)

    new_landmarks = f_landmarks(new_timesteps)
    new_mask = f_mask(new_timesteps) >= bad_detection_threshold

    new_landmarks[new_mask] = 0

    return new_landmarks


def change_duration(
    multi_frame_landmarks: Union[np.ndarray, torch.Tensor],
    factor: float = None,
    n_new_timesteps: int = None,
    time_dimention: int = 0,
    bad_detection_threshold: float = 0.1,
    use_random_speed: bool = False,
    random_factor_range: Tuple[float, float] = (0.5, 1.5),
    add_random_steps: bool = False,
    n_random_steps_frac: float = None,
    min_n_linear_steps: int = 20,
    min_n_linear_steps_frac: float = 0.25,
    add_slow_spots: bool = False,
    n_slow_spots: int = None,
    n_slow_spots_frac: float = 0.02,
    slow_spots_steps_frac: float = 0.2,
    return_new_timesteps: bool = False,
) -> Union[
    np.ndarray,
    torch.Tensor,
    Tuple[np.ndarray, np.ndarray],
    Tuple[torch.Tensor, torch.Tensor],
]:
    """increase or decrease the number of subarrays (frames) across a dimention (time) by interpolating new data within the given data.

    Args:
        multi_frame_landmarks (Union[np.ndarray, torch.Tensor]): a 3D array of landmarks to be interpolated
        factor (float, optional): multiplier of old duration to get new duration. Defaults to None.
        n_new_timesteps (int, optional): the new duration. Alternate to factor, overwrites that. Defaults to None.
        time_dimention (int, optional): the dimention number of the landmarks array across which interpolation should take place. Defaults to 0.
        bad_detection_threshold (float, optional): Used to convert interpolated mask into anther mask. values in interpolated mask greater than this are converted to True and corresponding landmarks put equal to 0. Defaults to 0.1.
        use_random_speed (bool, optional): determine a factor by random. only works if Nones are provided as factor/n_new_timesteps. Defaults to False.
        random_factor_range (Tuple[float, float], optional): uniformly sample a factor value from this range. Defaults to (0.5, 1.5).
        add_random_steps (bool, optional): True means sample some values from uniform distribution. Defaults to False.
        n_random_steps_frac (float, optional): what percentage of total steps should be sampled from uniform distribution. If None, the ratio is randomly selected. Defaults to None.
        min_n_linear_steps (int, optional): minimum number of steps that should be evenly spaced and span the entire range. Defaults to 20.
        min_n_linear_steps_frac (float, optional): percentage of total steps at least which should be evenly spaced and span the entire range. Its an alternate to the above and maximum of both is used. Defaults to 0.25.
        add_slow_spots (bool, optional): True means sample some values from a normal distribution concentrated around some point. Defaults to False.
        n_slow_spots (int, optional): number of such concentrated points to use. If None, its selected randomly. Defaults to None.
        n_slow_spots_frac (float, optional): ratio of number of spots to total steps. Defaults to 0.02.
        slow_spots_steps_frac (float, optional): percentage of total steps to be used to make such spots. Defaults to 0.2.
        return_new_timesteps (bool, optional): True means the new timesteps array to which the interpolation corresponds is also returned. Defaults to False.

    Raises:
        ValueError: speed factor, n_new_timesteps, and use_random_speed are all None/False

    Returns:
        np.ndarray | torch.Tensor | Tuple[np.ndarray, np.ndarray] | Tuple[torch.Tensor, torch.Tensor]: The interpolated landmarks and/or the timesteps they correspond to.
    """

    n_old_timesteps = multi_frame_landmarks.shape[time_dimention]

    old_timesteps = _prepare_timesteps(
        n_old_timesteps,
        return_type=type(multi_frame_landmarks),
        add_random_steps=False,
        add_slow_spots=False,
    )

    if n_new_timesteps is None:
        if factor is None:
            if use_random_speed:
                factor = (
                    np.random.rand(1)
                    * (random_factor_range[1] - random_factor_range[0])
                    + random_factor_range[0]
                )
            else:
                raise ValueError("provide a speed factor or n_new_timesteps")
        n_new_timesteps = int(n_old_timesteps * factor)

    new_timesteps = _prepare_timesteps(
        n_new_timesteps,
        return_type=type(multi_frame_landmarks),
        add_random_steps=add_random_steps,
        n_random_steps_frac=n_random_steps_frac,
        min_n_linear_steps=min_n_linear_steps,
        min_n_linear_steps_frac=min_n_linear_steps_frac,
        add_slow_spots=add_slow_spots,
        n_slow_spots=n_slow_spots,
        n_slow_spots_frac=n_slow_spots_frac,
        slow_spots_steps_frac=slow_spots_steps_frac,
    )

    new_landmarks = _interpolate_multi_frame_landmarks(
        multi_frame_landmarks,
        old_timesteps=old_timesteps,
        new_timesteps=new_timesteps,
        time_dimention=time_dimention,
        bad_detection_threshold=bad_detection_threshold,
    )

    return new_landmarks if not return_new_timesteps else (new_landmarks, new_timesteps)
