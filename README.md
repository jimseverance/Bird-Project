### yolov5 and birds

#### Setup

I used Anaconda 4.12.0, and Python 3.10.4

    conda activate py3104

##### Install Python modules
    
    pip install -r requirements.txt

##### Test using the built-in image and trained model

    yolov5 detect
    open runs/detect/exp/bus.jpg 

##### Run the bird species demo on an image
Using the default supplied jpeg file
On a Mac, you might have to ALT-Tab to the output process

    python find_birds.py --showImage --image ./media/blue-bird.jpeg -w 10000


Note that the demo runs **100x slower** on a Mac without a GPU. I.e. it runs like **100x faster** on a PC with CUDA and a 1080 Ti graphics card.

##### Run the bird species demo on a video

    python find-birds.py --showImage --image ./media/bird-feeder.mp4 

##### Project folder structure

- media - contains example image and video
- models - contains 
    the default yolov5 trained model - yolov5s.pt - trained on the coco dataset
    the custom bird-species trained model - birds.pt
- utils - utility/help code, copied from the yolov5 source code

##### Bird species training images 

https://www.kaggle.com/gpiosenka/100-bird-species

##### How to convert to darknet format? 

See kaggle-to-yolo5.py

##### Why reduce the training set to only 33 birds? https://

These are common birds found in Missour: http://www.whatbirdsareinmybackyard.com/2019/09/what-are-most-common-backyard-birds-in-missouri.html

Training is WAY faster - like 50x faster, when you use a reduced but still useful dataset.

##### How to train

yolov5 train --batch-size 16 --weights ./models/yolov5s.pt --data ./data/google-birds.yaml --epochs 20 --freeze 10

##### Yolo version

yolov7 is now out - https://github.com/akashAD98/yolov7-pip - I have not used it though

