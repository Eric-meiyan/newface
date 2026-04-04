from typing import Any, Dict, List, Optional, Tuple

import numpy

from facefusion import logger
from facefusion.types import AppContext, BoundingBox, FaceBlendshapes, FaceLandmark5, FacePoseMatrix, Score, VisionFrame

HAS_MEDIAPIPE = False

try:
	import mediapipe  # noqa: F401
	HAS_MEDIAPIPE = True
except ImportError:
	pass

LANDMARKER_POOL : Dict[AppContext, Any] = {}
DETECTOR_POOL : Dict[AppContext, Any] = {}


def check_mediapipe_available() -> bool:
	if not HAS_MEDIAPIPE:
		logger.error('MediaPipe is required for this model. Install with: pip install mediapipe', __name__)
		return False
	return True


def get_face_landmarker(model_path : str, app_context : AppContext = 'cli') -> Any:
	if not HAS_MEDIAPIPE:
		return None

	if app_context in LANDMARKER_POOL:
		return LANDMARKER_POOL[app_context]

	from mediapipe.tasks.python.core import base_options as mp_base_options
	from mediapipe.tasks.python.vision import face_landmarker as mp_face_landmarker
	from mediapipe.tasks.python.vision.core import vision_task_running_mode as mp_running_mode

	options = mp_face_landmarker.FaceLandmarkerOptions(
		base_options = mp_base_options.BaseOptions(model_asset_path = model_path),
		running_mode = mp_running_mode.VisionTaskRunningMode.IMAGE,
		output_face_blendshapes = True,
		output_facial_transformation_matrixes = True,
		num_faces = 1,
		min_face_detection_confidence = 0.5,
		min_face_presence_confidence = 0.5,
		min_tracking_confidence = 0.5
	)
	landmarker = mp_face_landmarker.FaceLandmarker.create_from_options(options)
	LANDMARKER_POOL[app_context] = landmarker
	return landmarker


def clear_face_landmarker(app_context : Optional[AppContext] = None) -> None:
	if app_context:
		landmarker = LANDMARKER_POOL.pop(app_context, None)
		if landmarker:
			landmarker.close()
	else:
		for context_landmarker in LANDMARKER_POOL.values():
			context_landmarker.close()
		LANDMARKER_POOL.clear()


def get_face_detector(model_path : str, app_context : AppContext = 'cli') -> Any:
	if not HAS_MEDIAPIPE:
		return None

	if app_context in DETECTOR_POOL:
		return DETECTOR_POOL[app_context]

	from mediapipe.tasks.python.core import base_options as mp_base_options
	from mediapipe.tasks.python.vision import face_detector as mp_face_detector
	from mediapipe.tasks.python.vision.core import vision_task_running_mode as mp_running_mode

	options = mp_face_detector.FaceDetectorOptions(
		base_options = mp_base_options.BaseOptions(model_asset_path = model_path),
		running_mode = mp_running_mode.VisionTaskRunningMode.IMAGE,
		min_detection_confidence = 0.5,
		min_suppression_threshold = 0.3
	)
	detector = mp_face_detector.FaceDetector.create_from_options(options)
	DETECTOR_POOL[app_context] = detector
	return detector


def clear_face_detector(app_context : Optional[AppContext] = None) -> None:
	if app_context:
		detector = DETECTOR_POOL.pop(app_context, None)
		if detector:
			detector.close()
	else:
		for context_detector in DETECTOR_POOL.values():
			context_detector.close()
		DETECTOR_POOL.clear()


def detect_faces(vision_frame : VisionFrame, model_path : str, app_context : AppContext = 'cli') -> Tuple[List[BoundingBox], List[Score], List[FaceLandmark5]]:
	detector = get_face_detector(model_path, app_context)
	bounding_boxes : List[BoundingBox] = []
	face_scores : List[Score] = []
	face_landmarks_5 : List[FaceLandmark5] = []

	if not detector:
		return bounding_boxes, face_scores, face_landmarks_5

	mp_image = create_mp_image(vision_frame)

	if mp_image is None:
		return bounding_boxes, face_scores, face_landmarks_5

	result = detector.detect(mp_image)
	frame_height, frame_width = vision_frame.shape[:2]

	for detection in result.detections:
		bbox = detection.bounding_box
		x_min = bbox.origin_x
		y_min = bbox.origin_y
		x_max = bbox.origin_x + bbox.width
		y_max = bbox.origin_y + bbox.height
		bounding_boxes.append(numpy.array([ x_min, y_min, x_max, y_max ], dtype = numpy.float64))
		face_scores.append(detection.categories[0].score if detection.categories else 0.0)
		keypoints = detection.keypoints

		if keypoints and len(keypoints) >= 6:
			left_eye = numpy.array([ keypoints[1].x * frame_width, keypoints[1].y * frame_height ])
			right_eye = numpy.array([ keypoints[0].x * frame_width, keypoints[0].y * frame_height ])
			nose_tip = numpy.array([ keypoints[2].x * frame_width, keypoints[2].y * frame_height ])
			mouth_center = numpy.array([ keypoints[3].x * frame_width, keypoints[3].y * frame_height ])
			mouth_width = (x_max - x_min) * 0.15
			left_mouth = mouth_center + numpy.array([ -mouth_width, 0 ])
			right_mouth = mouth_center + numpy.array([ mouth_width, 0 ])
			face_landmarks_5.append(numpy.array([ left_eye, right_eye, nose_tip, left_mouth, right_mouth ], dtype = numpy.float64))
		else:
			center_x = (x_min + x_max) / 2
			center_y = (y_min + y_max) / 2
			face_landmarks_5.append(numpy.array(
			[
				[ center_x - bbox.width * 0.15, center_y - bbox.height * 0.15 ],
				[ center_x + bbox.width * 0.15, center_y - bbox.height * 0.15 ],
				[ center_x, center_y ],
				[ center_x - bbox.width * 0.1, center_y + bbox.height * 0.15 ],
				[ center_x + bbox.width * 0.1, center_y + bbox.height * 0.15 ]
			], dtype = numpy.float64))

	return bounding_boxes, face_scores, face_landmarks_5


def create_mp_image(vision_frame : VisionFrame) -> Any:
	if not HAS_MEDIAPIPE:
		return None

	from mediapipe.python._framework_bindings import image as mp_image
	from mediapipe.python._framework_bindings import image_frame as mp_image_frame

	rgb_frame = vision_frame
	if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 3:
		rgb_frame = numpy.ascontiguousarray(rgb_frame)
	return mp_image.Image(image_format = mp_image_frame.ImageFormat.SRGB, data = rgb_frame)


def detect_landmarks(vision_frame : VisionFrame, model_path : str, app_context : AppContext = 'cli') -> Optional[Dict[str, Any]]:
	landmarker = get_face_landmarker(model_path, app_context)

	if not landmarker:
		return None

	mp_image = create_mp_image(vision_frame)

	if mp_image is None:
		return None

	result = landmarker.detect(mp_image)

	if not result.face_landmarks:
		return None

	frame_height, frame_width = vision_frame.shape[:2]
	landmarks = result.face_landmarks[0]
	landmark_468 = numpy.array(
		[ [ landmark.x * frame_width, landmark.y * frame_height, landmark.z * frame_width ] for landmark in landmarks[:468] ],
		dtype = numpy.float64
	)
	blendshapes : Optional[FaceBlendshapes] = None
	pose_matrix : Optional[FacePoseMatrix] = None

	if result.face_blendshapes:
		blendshapes = numpy.array(
			[ category.score for category in result.face_blendshapes[0] ],
			dtype = numpy.float32
		)

	if result.facial_transformation_matrixes:
		pose_matrix = numpy.array(result.facial_transformation_matrixes[0], dtype = numpy.float32)

	return\
	{
		'landmark_468': landmark_468,
		'blendshapes': blendshapes,
		'pose_matrix': pose_matrix
	}
