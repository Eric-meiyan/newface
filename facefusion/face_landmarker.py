from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy

from facefusion import inference_manager, mediapipe_manager, state_manager
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.face_helper import create_rotation_matrix_and_size, estimate_matrix_by_face_landmark_5, transform_points, warp_face_by_translation
from facefusion.filesystem import resolve_relative_path
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.types import Angle, BoundingBox, DownloadScope, DownloadSet, FaceLandmark5, FaceLandmark68, InferencePool, ModelSet, Prediction, Score, VisionFrame

MEDIAPIPE_468_TO_68 =\
[
	# jaw (17)
	162, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
	# left eyebrow (5)
	70, 63, 105, 66, 107,
	# right eyebrow (5)
	336, 296, 334, 293, 300,
	# nose bridge (4)
	168, 6, 197, 195,
	# nose tip (5)
	5, 4, 1, 19, 94,
	# left eye (6)
	33, 160, 158, 133, 153, 144,
	# right eye (6)
	362, 385, 387, 263, 373, 380,
	# outer lip (12)
	61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 181,
	# inner lip (8)
	78, 82, 13, 312, 308, 317, 14, 87
]

MEDIAPIPE_468_TO_5 =\
[
	# left eye center
	468,
	# right eye center
	473,
	# nose tip
	1,
	# left mouth corner
	61,
	# right mouth corner
	291
]


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'2dfan4':
		{
			'__metadata__':
			{
				'vendor': 'breadbread1984',
				'license': 'MIT',
				'year': 2018
			},
			'hashes':
			{
				'2dfan4':
				{
					'url': resolve_download_url('models-3.0.0', '2dfan4.hash'),
					'path': resolve_relative_path('../.assets/models/2dfan4.hash')
				}
			},
			'sources':
			{
				'2dfan4':
				{
					'url': resolve_download_url('models-3.0.0', '2dfan4.onnx'),
					'path': resolve_relative_path('../.assets/models/2dfan4.onnx')
				}
			},
			'size': (256, 256)
		},
		'peppa_wutz':
		{
			'__metadata__':
			{
				'vendor': 'Unknown',
				'license': 'Apache-2.0',
				'year': 2023
			},
			'hashes':
			{
				'peppa_wutz':
				{
					'url': resolve_download_url('models-3.0.0', 'peppa_wutz.hash'),
					'path': resolve_relative_path('../.assets/models/peppa_wutz.hash')
				}
			},
			'sources':
			{
				'peppa_wutz':
				{
					'url': resolve_download_url('models-3.0.0', 'peppa_wutz.onnx'),
					'path': resolve_relative_path('../.assets/models/peppa_wutz.onnx')
				}
			},
			'size': (256, 256)
		},
		'fan_68_5':
		{
			'__metadata__':
			{
				'vendor': 'FaceFusion',
				'license': 'OpenRAIL-M',
				'year': 2024
			},
			'hashes':
			{
				'fan_68_5':
				{
					'url': resolve_download_url('models-3.0.0', 'fan_68_5.hash'),
					'path': resolve_relative_path('../.assets/models/fan_68_5.hash')
				}
			},
			'sources':
			{
				'fan_68_5':
				{
					'url': resolve_download_url('models-3.0.0', 'fan_68_5.onnx'),
					'path': resolve_relative_path('../.assets/models/fan_68_5.onnx')
				}
			}
		},
		'mediapipe':
		{
			'__metadata__':
			{
				'vendor': 'Google',
				'license': 'Apache-2.0',
				'year': 2023
			},
			'path': resolve_relative_path('../.assets/models/face_landmarker_v2.task')
		}
	}


def get_inference_pool() -> InferencePool:
	face_landmarker_model = state_manager.get_item('face_landmarker_model')

	if face_landmarker_model == 'mediapipe':
		face_landmarker_model = 'many'

	model_names = [ face_landmarker_model, 'fan_68_5' ]
	_, model_source_set = collect_model_downloads()

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	face_landmarker_model = state_manager.get_item('face_landmarker_model')

	if face_landmarker_model == 'mediapipe':
		face_landmarker_model = 'many'

	model_names = [ face_landmarker_model, 'fan_68_5' ]
	inference_manager.clear_inference_pool(__name__, model_names)


def collect_model_downloads() -> Tuple[DownloadSet, DownloadSet]:
	model_set = create_static_model_set('full')
	model_hash_set : DownloadSet = {}
	model_source_set : DownloadSet = {}
	face_landmarker_model = state_manager.get_item('face_landmarker_model')
	if face_landmarker_model == 'mediapipe':
		face_landmarker_model = 'many'

	if face_landmarker_model != 'mediapipe':
		model_hash_set['fan_68_5'] = model_set.get('fan_68_5').get('hashes').get('fan_68_5')
		model_source_set['fan_68_5'] = model_set.get('fan_68_5').get('sources').get('fan_68_5')

	for model in [ '2dfan4', 'peppa_wutz' ]:
		if face_landmarker_model in [ 'many', model ]:
			model_hash_set[model] = model_set.get(model).get('hashes').get(model)
			model_source_set[model] = model_set.get(model).get('sources').get(model)

	return model_hash_set, model_source_set


def pre_check() -> bool:
	if state_manager.get_item('face_landmarker_model') == 'mediapipe':
		state_manager.set_item('face_landmarker_model', 'many')  # fallback for Python 3.13

	model_hash_set, model_source_set = collect_model_downloads()

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def detect_face_landmark(vision_frame : VisionFrame, bounding_box : BoundingBox, face_angle : Angle) -> Tuple[FaceLandmark68, Score, Optional[Dict[str, Any]]]:
	face_landmarker_model = state_manager.get_item('face_landmarker_model')

	if face_landmarker_model == 'mediapipe':
		face_landmarker_model = 'many'  # mediapipe not supported on Python 3.13, fallback to many

	face_landmark_2dfan4 = None
	face_landmark_peppa_wutz = None
	face_landmark_score_2dfan4 = 0.0
	face_landmark_score_peppa_wutz = 0.0

	if face_landmarker_model in [ 'many', '2dfan4' ]:
		face_landmark_2dfan4, face_landmark_score_2dfan4 = detect_with_2dfan4(vision_frame, bounding_box, face_angle)

	if face_landmarker_model in [ 'many', 'peppa_wutz' ]:
		face_landmark_peppa_wutz, face_landmark_score_peppa_wutz = detect_with_peppa_wutz(vision_frame, bounding_box, face_angle)

	if face_landmark_score_2dfan4 > face_landmark_score_peppa_wutz - 0.2:
		return face_landmark_2dfan4, face_landmark_score_2dfan4, None
	return face_landmark_peppa_wutz, face_landmark_score_peppa_wutz, None


def detect_with_mediapipe(vision_frame : VisionFrame, bounding_box : BoundingBox) -> Tuple[FaceLandmark68, Score, Optional[Dict[str, Any]]]:
	model_set = create_static_model_set('full').get('mediapipe')
	model_path = model_set.get('path')
	result = mediapipe_manager.detect_landmarks(vision_frame, model_path)

	if result is None:
		empty_landmark_68 = numpy.zeros((68, 2), dtype = numpy.float64)
		return empty_landmark_68, 0.0, None

	landmark_468 = result.get('landmark_468')
	face_landmark_68 = landmark_468[MEDIAPIPE_468_TO_68, :2]

	if len(landmark_468) >= 478:
		left_eye = landmark_468[MEDIAPIPE_468_TO_5[0], :2]
		right_eye = landmark_468[MEDIAPIPE_468_TO_5[1], :2]
	else:
		left_eye = numpy.mean(landmark_468[[ 33, 160, 158, 133, 153, 144 ], :2], axis = 0)
		right_eye = numpy.mean(landmark_468[[ 362, 385, 387, 263, 373, 380 ], :2], axis = 0)

	nose_tip = landmark_468[1, :2]
	left_mouth = landmark_468[61, :2]
	right_mouth = landmark_468[291, :2]
	face_landmark_5 = numpy.array([ left_eye, right_eye, nose_tip, left_mouth, right_mouth ], dtype = numpy.float64)

	mediapipe_data =\
	{
		'landmark_468': landmark_468,
		'landmark_5': face_landmark_5,
		'blendshapes': result.get('blendshapes'),
		'pose_matrix': result.get('pose_matrix')
	}
	return face_landmark_68, 1.0, mediapipe_data


def detect_with_2dfan4(temp_vision_frame: VisionFrame, bounding_box: BoundingBox, face_angle: Angle) -> Tuple[FaceLandmark68, Score]:
	model_size = create_static_model_set('full').get('2dfan4').get('size')
	scale = 195 / numpy.subtract(bounding_box[2:], bounding_box[:2]).max().clip(1, None)
	translation = (model_size[0] - numpy.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
	rotation_matrix, rotation_size = create_rotation_matrix_and_size(face_angle, model_size)
	crop_vision_frame, affine_matrix = warp_face_by_translation(temp_vision_frame, translation, scale, model_size)
	crop_vision_frame = cv2.warpAffine(crop_vision_frame, rotation_matrix, rotation_size)
	crop_vision_frame = conditional_optimize_contrast(crop_vision_frame)
	crop_vision_frame = crop_vision_frame.transpose(2, 0, 1).astype(numpy.float32) / 255.0
	face_landmark_68, face_heatmap = forward_with_2dfan4(crop_vision_frame)
	face_landmark_68 = face_landmark_68[:, :, :2][0] / 64 * 256
	face_landmark_68 = transform_points(face_landmark_68, cv2.invertAffineTransform(rotation_matrix))
	face_landmark_68 = transform_points(face_landmark_68, cv2.invertAffineTransform(affine_matrix))
	face_landmark_score_68 = numpy.amax(face_heatmap, axis = (2, 3))
	face_landmark_score_68 = numpy.mean(face_landmark_score_68)
	face_landmark_score_68 = numpy.interp(face_landmark_score_68, [ 0, 0.9 ], [ 0, 1 ])
	return face_landmark_68, face_landmark_score_68


def detect_with_peppa_wutz(temp_vision_frame : VisionFrame, bounding_box : BoundingBox, face_angle : Angle) -> Tuple[FaceLandmark68, Score]:
	model_size = create_static_model_set('full').get('peppa_wutz').get('size')
	scale = 195 / numpy.subtract(bounding_box[2:], bounding_box[:2]).max().clip(1, None)
	translation = (model_size[0] - numpy.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
	rotation_matrix, rotation_size = create_rotation_matrix_and_size(face_angle, model_size)
	crop_vision_frame, affine_matrix = warp_face_by_translation(temp_vision_frame, translation, scale, model_size)
	crop_vision_frame = cv2.warpAffine(crop_vision_frame, rotation_matrix, rotation_size)
	crop_vision_frame = conditional_optimize_contrast(crop_vision_frame)
	crop_vision_frame = crop_vision_frame.transpose(2, 0, 1).astype(numpy.float32) / 255.0
	crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis = 0)
	prediction = forward_with_peppa_wutz(crop_vision_frame)
	face_landmark_68 = prediction.reshape(-1, 3)[:, :2] / 64 * model_size[0]
	face_landmark_68 = transform_points(face_landmark_68, cv2.invertAffineTransform(rotation_matrix))
	face_landmark_68 = transform_points(face_landmark_68, cv2.invertAffineTransform(affine_matrix))
	face_landmark_score_68 = prediction.reshape(-1, 3)[:, 2].mean()
	face_landmark_score_68 = numpy.interp(face_landmark_score_68, [ 0, 0.95 ], [ 0, 1 ])
	return face_landmark_68, face_landmark_score_68


def conditional_optimize_contrast(crop_vision_frame : VisionFrame) -> VisionFrame:
	crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_RGB2Lab)
	if numpy.mean(crop_vision_frame[:, :, 0]) < 30: #type:ignore[arg-type]
		crop_vision_frame[:, :, 0] = cv2.createCLAHE(clipLimit = 2).apply(crop_vision_frame[:, :, 0])
	crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_Lab2RGB)
	return crop_vision_frame


def estimate_face_landmark_68_5(face_landmark_5 : FaceLandmark5) -> FaceLandmark68:
	affine_matrix = estimate_matrix_by_face_landmark_5(face_landmark_5, 'ffhq_512', (1, 1))
	face_landmark_5 = cv2.transform(face_landmark_5.reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
	face_landmark_68_5 = forward_fan_68_5(face_landmark_5)
	face_landmark_68_5 = cv2.transform(face_landmark_68_5.reshape(1, -1, 2), cv2.invertAffineTransform(affine_matrix)).reshape(-1, 2)
	return face_landmark_68_5


def forward_with_2dfan4(crop_vision_frame : VisionFrame) -> Tuple[Prediction, Prediction]:
	face_landmarker = get_inference_pool().get('2dfan4')

	with conditional_thread_semaphore():
		prediction = face_landmarker.run(None,
		{
			'input': [ crop_vision_frame ]
		})

	return prediction


def forward_with_peppa_wutz(crop_vision_frame : VisionFrame) -> Prediction:
	face_landmarker = get_inference_pool().get('peppa_wutz')

	with conditional_thread_semaphore():
		prediction = face_landmarker.run(None,
		{
			'input': crop_vision_frame
		})[0]

	return prediction


def forward_fan_68_5(face_landmark_5 : FaceLandmark5) -> FaceLandmark68:
	face_landmarker = get_inference_pool().get('fan_68_5')

	with conditional_thread_semaphore():
		face_landmark_68_5 = face_landmarker.run(None,
		{
			'input': [ face_landmark_5 ]
		})[0][0]

	return face_landmark_68_5
