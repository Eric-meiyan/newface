import os
import subprocess
import threading
from typing import Iterator, Optional

import cv2
import numpy
from tqdm import tqdm

from facefusion import ffmpeg_builder, logger, state_manager, translator
from facefusion.audio import create_empty_audio_frame
from facefusion.content_analyser import analyse_stream
from facefusion.ffmpeg import open_ffmpeg
from facefusion.filesystem import is_directory
from facefusion.processors.core import get_processors_modules
from facefusion.types import Fps, StreamMode, VisionFrame
from facefusion.vision import extract_vision_mask, read_static_images


def multi_process_capture(camera_capture : cv2.VideoCapture, camera_fps : Fps, stream_process_scale : float = 1.0, stream_frame_drop : bool = True) -> Iterator[VisionFrame]:
	latest_frame : Optional[VisionFrame] = None
	frame_lock = threading.Lock()
	is_running = True

	def capture_loop() -> None:
		nonlocal latest_frame, is_running

		while is_running and camera_capture and camera_capture.isOpened():
			ret, frame = camera_capture.read()

			if not ret or not numpy.any(frame):
				continue

			if analyse_stream(frame, camera_fps):
				camera_capture.release()
				is_running = False
				return

			if stream_frame_drop:
				with frame_lock:
					latest_frame = frame
			else:
				with frame_lock:
					latest_frame = frame

	capture_thread = threading.Thread(target = capture_loop, daemon = True)
	capture_thread.start()

	with tqdm(desc = translator.get('streaming'), unit = 'frame', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
		while is_running:
			with frame_lock:
				frame = latest_frame
				latest_frame = None

			if frame is None:
				continue

			processed_frame = process_stream_frame(frame, stream_process_scale)
			progress.update()
			yield processed_frame

	capture_thread.join(timeout = 5.0)


def process_stream_frame(target_vision_frame : VisionFrame, stream_process_scale : float = 1.0) -> VisionFrame:
	if stream_process_scale < 1.0:
		original_height, original_width = target_vision_frame.shape[:2]
		scaled_width = int(original_width * stream_process_scale)
		scaled_height = int(original_height * stream_process_scale)
		scaled_width = scaled_width + scaled_width % 2
		scaled_height = scaled_height + scaled_height % 2
		process_frame = cv2.resize(target_vision_frame, (scaled_width, scaled_height))
	else:
		process_frame = target_vision_frame
		original_height, original_width = 0, 0

	source_vision_frames = read_static_images(state_manager.get_item('source_paths'))
	source_audio_frame = create_empty_audio_frame()
	source_voice_frame = create_empty_audio_frame()
	temp_vision_frame = process_frame.copy()
	temp_vision_mask = extract_vision_mask(temp_vision_frame)

	for processor_module in get_processors_modules(state_manager.get_item('processors')):
		logger.disable()
		if processor_module.pre_process('stream'):
			logger.enable()
			temp_vision_frame, temp_vision_mask = processor_module.process_frame(
			{
				'source_vision_frames': source_vision_frames,
				'source_audio_frame': source_audio_frame,
				'source_voice_frame': source_voice_frame,
				'target_vision_frame': process_frame,
				'temp_vision_frame': temp_vision_frame,
				'temp_vision_mask': temp_vision_mask
			})
		logger.enable()

	if stream_process_scale < 1.0:
		temp_vision_frame = cv2.resize(temp_vision_frame, (original_width, original_height))

	return temp_vision_frame


def open_stream(stream_mode : StreamMode, stream_resolution : str, stream_fps : Fps) -> subprocess.Popen[bytes]:
	commands = ffmpeg_builder.chain(
		ffmpeg_builder.capture_video(),
		ffmpeg_builder.set_media_resolution(stream_resolution),
		ffmpeg_builder.set_input_fps(stream_fps)
	)

	if stream_mode == 'udp':
		commands.extend(ffmpeg_builder.set_input('-'))
		commands.extend(ffmpeg_builder.set_stream_mode('udp'))
		commands.extend(ffmpeg_builder.set_stream_quality(2000))
		commands.extend(ffmpeg_builder.set_output('udp://localhost:27000?pkt_size=1316'))

	if stream_mode == 'v4l2':
		device_directory_path = '/sys/devices/virtual/video4linux'
		commands.extend(ffmpeg_builder.set_input('-'))
		commands.extend(ffmpeg_builder.set_stream_mode('v4l2'))

		if is_directory(device_directory_path):
			device_names = os.listdir(device_directory_path)

			for device_name in device_names:
				device_path = '/dev/' + device_name
				commands.extend(ffmpeg_builder.set_output(device_path))

		else:
			logger.error(translator.get('stream_not_loaded').format(stream_mode = stream_mode), __name__)

	return open_ffmpeg(commands)
