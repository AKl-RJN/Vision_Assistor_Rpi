
import webcamvideostream

class VideoStream:
	def __init__(self, src=0, usePiCamera=False, resolution=(320, 240),
		framerate=32, **kwargs):
		
		if usePiCamera:
			
			import pivideoStream

			# initialize the picamera stream 
			self.stream = PiVideoStream(resolution=resolution,
				framerate=framerate, **kwargs)

		# or we use the webcam
		
		else:
			self.stream = WebcamVideoStream(src=src)

	def start(self):
		
		return self.stream.start()

	def update(self):
		
		self.stream.update()

	def read(self):
		# return the current frame
		return self.stream.read()

	def stop(self):
		
		self.stream.stop()
