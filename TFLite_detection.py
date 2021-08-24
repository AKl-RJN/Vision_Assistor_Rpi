
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from picamera import PiCamera
from gtts import gTTS
import pygame
language = 'en'

from pivideostream import PiVideoStream
from imutils.video.videostream import VideoStream



"""
class VideoStream:
    Camera object that controls video streaming from the Picamera
    def __init__(self,resolution=(640,480),usePiCamera=True,framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = PiVideoStream(resolution=resolution,framerate=framerate)
        #ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        #ret = self.stream.set(3,resolution[0])
        #ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        #(self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            #(self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True
        """

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')



args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu


pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate


if use_TPU:
   
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]


if labels[0] == '???':
    del(labels[0])


if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(usePiCamera=True,framerate=30).start()
time.sleep(1)


while True:

    
    t1 = cv2.getTickCount()

   
    frame1 = videostream.read()

    
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

   
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
   

    
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            
            object_name = labels[int(classes[i])]
            mytext = str(object_name)
     
            if mytext == "person":
                pygame.mixer.init()
                pygame.mixer.music.load("person.mp3")
                pygame.mixer.music.play()
            if mytext == "bicycle":
                pygame.mixer.init()
                pygame.mixer.music.load("bicycle.mp3")
                pygame.mixer.music.play()
            if mytext == "car":
                pygame.mixer.init()
                pygame.mixer.music.load("car.mp3")
                pygame.mixer.music.play()
            if mytext == "motorcycle":
                pygame.mixer.init()
                pygame.mixer.music.load("motorcycle.mp3")
                pygame.mixer.music.play()
            if mytext == "bus":
                pygame.mixer.init()
                pygame.mixer.music.load("bus.mp3")
                pygame.mixer.music.play()
            if mytext == "train":
                pygame.mixer.init()
                pygame.mixer.music.load("train.mp3")
                pygame.mixer.music.play()
            if mytext == "truck":
                pygame.mixer.init()
                pygame.mixer.music.load("truck.mp3")
                pygame.mixer.music.play()
            if mytext == "traffic light":
                pygame.mixer.init()
                pygame.mixer.music.load("traffic light.mp3")
                pygame.mixer.music.play()
            if mytext == "door":
                pygame.mixer.init()
                pygame.mixer.music.load("door.mp3")
                pygame.mixer.music.play()
            if mytext == "chair":
                pygame.mixer.init()
                pygame.mixer.music.load("chair.mp3")
                pygame.mixer.music.play()
            if mytext == "couch":
                pygame.mixer.init()
                pygame.mixer.music.load("couch.mp3")
                pygame.mixer.music.play()
            if mytext == "potted plant":
                pygame.mixer.init()
                pygame.mixer.music.load("potted plant.mp3")
                pygame.mixer.music.play()
            
            if mytext == "cow":
                pygame.mixer.init()
                pygame.mixer.music.load("cow.mp3")
                pygame.mixer.music.play()
            if mytext == "elephant":
                pygame.mixer.init()
                pygame.mixer.music.load("elephant.mp3")
                pygame.mixer.music.play()
            
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) 
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
            label_ymin = max(ymin, labelSize[1] + 10) 
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
            cv2.putText(frame,label,(xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 

    
    cv2.putText(frame,'FPS:{0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()
videostream.stop()
