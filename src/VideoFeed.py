import cv2, queue, threading, numpy
from src.Helpers import pil2opencv

class VideoFeed:
    def __init__(self, camera, display=True):
        print(' - OPENING CAMERA FEED')
        self.feed = cv2.VideoCapture(camera)
        self.queue = queue.Queue()
        self.displayImg = None
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()
        if display:
            t2 = threading.Thread(target=self._display)
            t2.daemon = True
            t2.start()
        print(' | - Camera opened correctly')

    def _reader(self):
        while True:
            ret, frame = self.feed.read()
            if not ret:
                break
            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(frame)

    def _display(self):
        while True:
            if self.displayImg is not None:
                cv2.imshow('Detection', self.displayImg)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    def read(self):
        return cv2.cvtColor(self.queue.get(), cv2.COLOR_BGR2RGB)

    def display(self, img):
        self.displayImg = pil2opencv(img)

    def release(self):
        self.feed.release()
