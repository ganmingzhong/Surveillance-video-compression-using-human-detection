import torch
import numpy as np
import cv2
from time import time

import os
import imageio
import math

#%% Class for obj detection

class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using OpenCV.
    """
    
    def __init__(self):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        #self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        #self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def get_class_list(self):
        return self.classes


    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def runDetection(self, frame):
      results = self.score_frame(frame)
      frame = self.plot_boxes(results, frame)
      return frame, results

from matplotlib import pyplot as plt
path = 'C:/Users/User/Desktop/output_frame2'
def summarizeVideo(videoFile, duration):
    
    cap = cv2.VideoCapture(videoFile)
    framesList=[]
    count=0
    obj = ObjectDetection()
    #here
    current_no_person=0
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_ori=frame
        
        if ret==True:
            img_out, results_out = obj.runDetection(frame)
            labels, cord = results_out
            class_list = obj.get_class_list()
            n = len(labels)
            obj_list=[]
                #out.write(frame)
            for i in range(n):
                obj_list.append(obj.class_to_label(labels[i]))
            previous_no_person=current_no_person
            
            if "person" in obj_list:
                current_no_person=obj_list.count("person")
                if current_no_person>previous_no_person+1 or current_no_person<previous_no_person-1:
                    cv2.imwrite("frame%d.jpg" % count, img_out)
                    #print(img_out)
                    framesList.append("frame%d.jpg" % count)
                    #cv2.imwrite(os.path.join(path, 'frame_ori%d.jpg'% count), frame_ori)
                    cv2.imwrite(os.path.join(path , 'frame%d.jpg'% count), img_out)
                    count+=1 
                
        else:
            break
            

    cap.release()


    return framesList


# Main code
fig=plt.figure()

videoFile = 'surveillance_5_short.mp4'
framesList=summarizeVideo(videoFile,10) #10 minutes
# Visualize the first 20 frames

for i, imf in enumerate(framesList[:20]):
    fig.set_figheight(50)
    fig.set_figwidth(50)
    fig.add_subplot(10, 2, 1+i)
    im=cv2.imread(imf)
    plt.imshow(im)
    plt.axis('off')
