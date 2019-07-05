import cv2, queue, threading

class VideoFeed:
    def __init__(self, camera):
        print(' - OPENING CAMERA FEED')
        self.feed = cv2.VideoCapture(camera)
        self.queue = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()
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

    def read(self):
        return cv2.cvtColor(self.queue.get(), cv2.COLOR_BGR2RGB)

    def release(self):
        self.feed.release()
