# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Perform test request
"""

import pprint
import requests
import argparse
import cv2
import numpy as np
from PIL import Image
import io
import yolov5
import json
import imutils

from utils.plots import Annotator, colors, save_one_box

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end

def predict(pil_image):
        # Convert to cv2
        open_cv_image = np.array(pil_image) 

        # Convert RGB to BGR 
        img = open_cv_image.copy() 

        results = model(img, size=640)  # reduce size=320 for faster inference

        xyxy = results.pandas().xyxy[0]

        if birdsOnly:
            xyxy = xyxy.loc[xyxy['name'] == 'bird']

        for index, prediction in xyxy.iterrows():

        # ??? Would the inference work faster if the all image patches were passed in a single inference call?
        # for index, bird_prediction in birds.iterrows():
            
            # If it's not a bird, don't try to predict bird species
            if prediction['name'] != 'bird':
                continue

            # # Bounding box coordinates
            box = prediction[ :4] # x1, y1, x2, y2

            # Create a new image containing the contents of the bounding box
            roi = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

            # Perform bird-species-specific inference
            results2 = model2(roi)
            
            # Grab the inference results
            predictions2 = results2.pred[0]

            # If no prediction is made, leave the 'bird' label alone
            if predictions2.size(dim=0) == 0:
                continue

            categories2 = predictions2[0, 5]

            # Replace the generic 'bird' label and confidence with the bird species label and confidence
            xyxy.at[index,'name'] = names2[int(categories2)]
            
            if debug:
                print(xyxy.to_json(orient="records"))
                
        # return birds.to_json(orient="records")
        return xyxy.to_json(orient="records")


def image_to_byte_array(image: Image) -> bytes:
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='jpeg')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr


ap = argparse.ArgumentParser()
ap.add_argument("--url", required=False, default='http://localhost:5000/v1/object-detection/yolov5s',
	help="URL of the service")
ap.add_argument("--image", required=False, default='./media/blue-bird.jpeg',
	help="filespec of the image to process")
ap.add_argument('-d', '--debug', action='store_true', help='debug')
ap.add_argument('--showImage', action='store_true', help='debug')
ap.add_argument('-b','--birdsOnly', action='store_true', help='detect only birds')
ap.add_argument('-w', '--waitTime', type=int, default=1, help='Wait time for displaying image')
ap.add_argument("--cocoModel", default='./models/yolov5s.pt',  help="coco model filespec")
ap.add_argument("--birdModel", default='./models/birds.pt',  help="bird model filespec")
ap.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
ap.add_argument('-m', '--minConfidence', type=float, choices=[Range(0.0, 1.0)], default=0.3, help='Minimum confidence level')

args = vars(ap.parse_args())
debug = args["debug"]
birdsOnly = args["birdsOnly"]
minConfidence = args["minConfidence"]
waitTime = args["waitTime"]

# load pretrained coco model
model = yolov5.load(args["cocoModel"])

# load custom bird model
model2 = yolov5.load(args["birdModel"])

# The names (ascii labels) that the model was trained on
names2 = model2.names

# # Read image or video from file as a blob

cap = cv2.VideoCapture(args["image"])
# Check if image/file/camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

# Read until input is completed.
while(cap.isOpened()):
  ret, image_data = cap.read()
  if ret == True:

    cv2_im = cv2.cvtColor(image_data,cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_im)

    response = predict(pil_image)
    response = json.loads(response)

    # Convert the raw file bytes into an opencv-format image so it can be annotated
    open_cv_image = np.array(pil_image) 

    # Annotate the *original* image with the labels, if any, that were found
    annotator = Annotator(open_cv_image)

    for detection in response:
        print(f'detection: {detection}')

        confidence = float(detection['confidence'])
        if confidence < minConfidence:
                continue

        box = [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']]
        annotator.box_label(box, label=detection['name'])

    img = annotator.im

    # Debug - display the image, wait for a response
    if args["showImage"]:

        # Downsize and maintain aspect ratio so the window is small enough to fit on the monitor without scrolling
        img2 = imutils.resize(img, width=800)

        cv2.imshow('im',cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(waitTime) & 0xFF == ord('q'):
            break

    if args["debug"]:
        pprint.pprint(response)

  # Break the loop
  else: 
    break

