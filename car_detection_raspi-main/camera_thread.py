import logging
import queue
import threading
import time

import cv2

class CameraThread:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize camera
        self.camera_source = 0
        self.camera_width = 640
        self.camera_height = 480
        self.camera_frame_rate = 30
        self.camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        self.camera = None
        self.buffer = queue.Queue(1)
        self.frame_grab_run = False
        self.frame_grab_on = False
        self.frame_count = 0
        self.frames_returned = 0
        self.current_frame_rate = 0
        self.loop_start_time = 0
        self.thread = None

    def start(self):
        self.camera = cv2.VideoCapture(self.camera_source)

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.camera.set(cv2.CAP_PROP_FPS, self.camera_frame_rate)
        self.camera.set(cv2.CAP_PROP_FOURCC, self.camera_fourcc)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        time.sleep(0.5)  # Allow camera to initialize

        if self.camera.isOpened():
            self.frame_grab_run = True
            self.thread = threading.Thread(target=self.loop)
            self.thread.start()
        else:
            self.logger.error("camera could not be opened.")

    def stop(self):
        self.frame_grab_run = False
        if self.thread:
            self.thread.join()

        if self.camera:
            self.camera.release()
        self.buffer = None

    def loop(self):
        self.frame_grab_on = True
        while self.frame_grab_run:
            ret, frame = self.camera.read()
            if not ret:
                break
            if self.buffer.full():
                try:
                    self.buffer.get(block=False)
                    self.frame_count -= 1
                except queue.Empty:
                    pass
            self.buffer.put(frame)
            self.frame_count += 1
        self.frame_grab_on = False

    def next(self):
        frame = None
        try:
            frame = self.buffer.get()
            self.frames_returned += 1
        except queue.Empty:
            pass
        return frame
    
    def is_next(self):
        return self.buffer.full()