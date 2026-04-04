from typing import Tuple

import cv2
import numpy

from facefusion.types import FacePoseMatrix, VisionFrame


def decompose_pose_matrix(pose_matrix : FacePoseMatrix) -> Tuple[float, float, float, numpy.ndarray]:
	rotation_matrix = pose_matrix[:3, :3]
	scale = numpy.linalg.norm(rotation_matrix[:, 0])

	if scale > 0:
		rotation_matrix = rotation_matrix / scale

	sy = numpy.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
	is_singular = sy < 1e-6

	if not is_singular:
		pitch = numpy.degrees(numpy.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
		yaw = numpy.degrees(numpy.arctan2(-rotation_matrix[2, 0], sy))
		roll = numpy.degrees(numpy.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
	else:
		pitch = numpy.degrees(numpy.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
		yaw = numpy.degrees(numpy.arctan2(-rotation_matrix[2, 0], sy))
		roll = 0.0

	translation = pose_matrix[:3, 3]
	return float(pitch), float(yaw), float(roll), translation


def compensate_perspective(crop_vision_frame : VisionFrame, source_pose_matrix : FacePoseMatrix, target_pose_matrix : FacePoseMatrix) -> VisionFrame:
	_, source_yaw, _, _ = decompose_pose_matrix(source_pose_matrix)
	_, target_yaw, _, _ = decompose_pose_matrix(target_pose_matrix)
	yaw_diff = target_yaw - source_yaw
	frame_height, frame_width = crop_vision_frame.shape[:2]
	center_x = frame_width / 2
	center_y = frame_height / 2
	focal_length = frame_width
	yaw_rad = numpy.radians(yaw_diff)
	cos_yaw = numpy.cos(yaw_rad)
	sin_yaw = numpy.sin(yaw_rad)
	rotation_3d = numpy.array(
	[
		[ cos_yaw, 0, sin_yaw ],
		[ 0, 1, 0 ],
		[ -sin_yaw, 0, cos_yaw ]
	], dtype = numpy.float64)
	camera_matrix = numpy.array(
	[
		[ focal_length, 0, center_x ],
		[ 0, focal_length, center_y ],
		[ 0, 0, 1 ]
	], dtype = numpy.float64)
	perspective_matrix = camera_matrix @ rotation_3d @ numpy.linalg.inv(camera_matrix)
	compensated_frame = cv2.warpPerspective(crop_vision_frame, perspective_matrix, (frame_width, frame_height), borderMode = cv2.BORDER_REPLICATE)
	return compensated_frame
