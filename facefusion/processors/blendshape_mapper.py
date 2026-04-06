import numpy

from facefusion.processors.types import LivePortraitExpression
from facefusion.types import FaceBlendshapes

# MediaPipe blendshape indices (52 total, index 0 is _neutral)
# https://developers.google.com/mediapipe/solutions/vision/face_landmarker/index
BS_BROW_DOWN_LEFT = 1
BS_BROW_DOWN_RIGHT = 2
BS_BROW_INNER_UP = 3
BS_BROW_OUTER_UP_LEFT = 4
BS_BROW_OUTER_UP_RIGHT = 5
BS_EYE_BLINK_LEFT = 9
BS_EYE_BLINK_RIGHT = 10
BS_EYE_LOOK_DOWN_LEFT = 11
BS_EYE_LOOK_DOWN_RIGHT = 12
BS_EYE_LOOK_IN_LEFT = 13
BS_EYE_LOOK_IN_RIGHT = 14
BS_EYE_LOOK_OUT_LEFT = 15
BS_EYE_LOOK_OUT_RIGHT = 16
BS_EYE_LOOK_UP_LEFT = 17
BS_EYE_LOOK_UP_RIGHT = 18
BS_EYE_SQUINT_LEFT = 19
BS_EYE_SQUINT_RIGHT = 20
BS_EYE_WIDE_LEFT = 21
BS_EYE_WIDE_RIGHT = 22
BS_JAW_FORWARD = 23
BS_JAW_LEFT = 24
BS_JAW_OPEN = 25
BS_JAW_RIGHT = 26
BS_MOUTH_CLOSE = 27
BS_MOUTH_FUNNEL = 28
BS_MOUTH_LEFT = 30
BS_MOUTH_LOWER_DOWN_LEFT = 32
BS_MOUTH_LOWER_DOWN_RIGHT = 33
BS_MOUTH_PRESS_LEFT = 34
BS_MOUTH_PRESS_RIGHT = 35
BS_MOUTH_PUCKER = 36
BS_MOUTH_RIGHT = 37
BS_MOUTH_ROLL_LOWER = 38
BS_MOUTH_ROLL_UPPER = 39
BS_MOUTH_SHRINK_LOWER = 40
BS_MOUTH_SHRINK_UPPER = 41
BS_MOUTH_SMILE_LEFT = 44
BS_MOUTH_SMILE_RIGHT = 45
BS_MOUTH_STRETCH_LEFT = 46
BS_MOUTH_STRETCH_RIGHT = 47
BS_MOUTH_UPPER_UP_LEFT = 48
BS_MOUTH_UPPER_UP_RIGHT = 49
BS_NOSE_SNEER_LEFT = 50
BS_NOSE_SNEER_RIGHT = 51

# LivePortrait expression coefficient indices (21 total, each has xyz)
# Grouped by face region:
#   Always fixed: [0, 4, 5, 8, 9]
#   Upper face:   [1, 2, 6, 10, 11, 12, 13, 15, 16]
#   Lower face:   [3, 7, 14, 17, 18, 19, 20]

LP_FIXED = [ 0, 4, 5, 8, 9 ]
LP_UPPER = [ 1, 2, 6, 10, 11, 12, 13, 15, 16 ]
LP_LOWER = [ 3, 7, 14, 17, 18, 19, 20 ]

# Expression range from live_portrait.py for normalization reference
EXPRESSION_SCALE = 0.05


def map_blendshapes_to_expression(blendshapes : FaceBlendshapes) -> LivePortraitExpression:
	expression = numpy.zeros((1, 21, 3), dtype = numpy.float32)

	if blendshapes is None or len(blendshapes) < 52:
		return expression

	# Upper face: eye blinks -> coefficients 1, 2
	blink_left = blendshapes[BS_EYE_BLINK_LEFT]
	blink_right = blendshapes[BS_EYE_BLINK_RIGHT]
	expression[0, 1, 0] = -blink_left * EXPRESSION_SCALE
	expression[0, 1, 1] = -blink_left * EXPRESSION_SCALE * 0.5
	expression[0, 2, 0] = -blink_right * EXPRESSION_SCALE
	expression[0, 2, 1] = -blink_right * EXPRESSION_SCALE * 0.5

	# Upper face: eye wide -> coefficients 6
	wide_left = blendshapes[BS_EYE_WIDE_LEFT]
	wide_right = blendshapes[BS_EYE_WIDE_RIGHT]
	expression[0, 6, 0] = (wide_left + wide_right) * 0.5 * EXPRESSION_SCALE * 0.3

	# Upper face: brow -> coefficients 10, 11, 12
	brow_down = (blendshapes[BS_BROW_DOWN_LEFT] + blendshapes[BS_BROW_DOWN_RIGHT]) * 0.5
	brow_up = blendshapes[BS_BROW_INNER_UP]
	brow_outer_left = blendshapes[BS_BROW_OUTER_UP_LEFT]
	brow_outer_right = blendshapes[BS_BROW_OUTER_UP_RIGHT]
	expression[0, 10, 0] = (brow_up - brow_down) * EXPRESSION_SCALE
	expression[0, 10, 2] = (brow_outer_left - brow_outer_right) * EXPRESSION_SCALE * 0.5
	expression[0, 11, 0] = (brow_up - brow_down) * EXPRESSION_SCALE * 0.5
	expression[0, 12, 0] = brow_up * EXPRESSION_SCALE * 0.3

	# Upper face: eye squint -> coefficients 13, 15, 16
	squint = (blendshapes[BS_EYE_SQUINT_LEFT] + blendshapes[BS_EYE_SQUINT_RIGHT]) * 0.5
	expression[0, 13, 0] = -squint * EXPRESSION_SCALE * 0.3
	expression[0, 15, 0] = -squint * EXPRESSION_SCALE * 0.2
	expression[0, 16, 0] = -squint * EXPRESSION_SCALE * 0.2

	# Lower face: jaw open -> coefficients 3, 7
	jaw_open = blendshapes[BS_JAW_OPEN]
	expression[0, 3, 0] = -jaw_open * EXPRESSION_SCALE
	expression[0, 3, 1] = -jaw_open * EXPRESSION_SCALE * 0.5
	expression[0, 7, 0] = jaw_open * EXPRESSION_SCALE * 0.8
	expression[0, 7, 1] = jaw_open * EXPRESSION_SCALE * 0.3

	# Lower face: jaw lateral -> coefficient 14
	jaw_lateral = blendshapes[BS_JAW_LEFT] - blendshapes[BS_JAW_RIGHT]
	expression[0, 14, 0] = jaw_lateral * EXPRESSION_SCALE * 0.3

	# Lower face: mouth -> coefficients 17, 18, 19, 20
	smile = (blendshapes[BS_MOUTH_SMILE_LEFT] + blendshapes[BS_MOUTH_SMILE_RIGHT]) * 0.5
	pucker = blendshapes[BS_MOUTH_PUCKER]
	mouth_open = blendshapes[BS_MOUTH_CLOSE]
	mouth_left = blendshapes[BS_MOUTH_LEFT] - blendshapes[BS_MOUTH_RIGHT]

	expression[0, 17, 0] = -smile * EXPRESSION_SCALE * 0.3
	expression[0, 17, 2] = smile * EXPRESSION_SCALE * 0.2
	expression[0, 18, 0] = pucker * EXPRESSION_SCALE * 0.3
	expression[0, 19, 0] = -smile * EXPRESSION_SCALE
	expression[0, 19, 1] = smile * EXPRESSION_SCALE * 1.5
	expression[0, 19, 2] = smile * EXPRESSION_SCALE * 0.5
	expression[0, 20, 0] = mouth_open * EXPRESSION_SCALE * 0.3
	expression[0, 20, 1] = mouth_left * EXPRESSION_SCALE * 0.2

	return expression


def blend_expressions(target_expression : LivePortraitExpression, temp_expression : LivePortraitExpression, factor : float) -> LivePortraitExpression:
	return target_expression * factor + temp_expression * (1 - factor)
