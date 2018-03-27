#!/usr/bin/python
# The contents of this file are in the public domain. See
LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
# This example program shows how to find frontal human faces in an image and
# estimate their pose. The pose takes the form of 68 landmarks. These are
# points on the face such as the corners of the mouth, along the eyebrows, on
# the eyes, and so forth.
#
# This face detector is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme. The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset.
#
# Also, note that you can train your own models using dlib's machine learning
# tools. See train_shape_predictor.py to see an example.
#
# You can get the shape_predictor_68_face_landmarks.dat file from:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
# You can install dlib using the command:
# pip install dlib
#
# Alternatively, if you want to compile dlib yourself then go into the dlib
# root folder and run:
# python setup.py install
 [AUTOMATIC CARICATURE GENERATION]
42
# or
# python setup.py install --yes USE_AVX_INSTRUCTIONS
# if you have a CPU that supports AVX instructions, since this makes some
# things run faster.
#
# Compiling dlib should work on any operating system so long as you have
# CMake and boost-python installed. On Ubuntu, this can be done easily by
# running the command:
# sudo apt-get install libboost-python-dev cmake
#
# Also note that this example requires scikit-image which can be installed
# via the command:
# pip install scikit-image
# Or downloaded from http://scikit-image.org/download.html.
import sys
import os
import dlib
import glob
import turtle
import time
import numpy as np
import argparse
import cv2
import threading
from skimage import io
import tkinter as tk
import tkinter
from PIL import ImageTk, Image
def helloCallBack():
 print('hello')
def clearf():
 turtle.reset()
yvalues=[423,545,605,663,720,771,815,849,858,848,814,772,719,662,603,543,481,449,4
 [AUTOMATIC CARICATURE GENERATION]
43
33,429,437,454,454,437,430,436,455,490,529,569,610,632,640,646,639,632,490,478,479
,495,502,503,497,479,479,491,504,504,716,698,684,690,685,697,715,736,744,746,744,7
36,716,709,710,709,715,708,710];
xvalues=[323,320,341,356,375,406,449,501,558,616,669,712,745,763,778,791,799,364,3
98,438,479,518,600,640,603,724,757,559,559,559,559,520,538,559,580,598,412,437,469
,495,465,435,622,647,679,705,881,650,486,514,540,559,577,604,631,605,578,558,538,5
13,502,539,559,577,615,577,558];
#predictor_path = sys.argv[1]
predictor_path = "shape_predictor_68_face_landmarks.dat"
#faces_folder_path = sys.argv[2]
faces_folder_path = "../examples/faces"
print("predictor: ",predictor_path)
xlimit1=0
ylimit1=0
xlimit2=1000
ylimit2=1000
print("facefolder: ",faces_folder_path)
n=sys.argv[1]
pth=faces_folder_path + "/" + n +".jpg"
print(pth)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
 t1 = threading.Thread(target=clearf())
 t1.start()
 t1.join()

 print("Processing file: {}".format(f))
 img = io.imread(pth)
 image = cv2.imread(pth)
 [AUTOMATIC CARICATURE GENERATION]
44
 boundaries = [
([14,14,16],[100,100,100])
]
 for (lower, upper) in boundaries:
 lower = np.array(lower, dtype = "uint8")
 upper = np.array(upper, dtype = "uint8")
 mask = cv2.inRange(image, lower, upper)
 output = cv2.bitwise_and(image, image, mask = mask)
 grayscaled = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
 retval, threshold = cv2.threshold(grayscaled, 10, 255, cv2.THRESH_BINARY)
 #cv2.imshow('threshold',threshold)

 win.clear_overlay()
 win.set_image(img)
 # Ask the detector to find the bounding boxes of each face. The 1 in the
 # second argument indicates that we should upsample the image 1 time. This
 # will make everything bigger and allow us to detect more faces.
 dets = detector(img, 1)
 print("Number of faces detected: {}".format(len(dets)))
 for k, d in enumerate(dets):

 print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
 k, d.left(), d.top(), d.right(), d.bottom()))
 # Get the landmarks/parts for the
 [AUTOMATIC CARICATURE GENERATION]
45
 turtle.setup( width = 500, height = 500, startx = None, starty = None)

 print("cordinates",shape.part(0))
 turtle.penup()
 #turtle.setpos((-shape.part(0).x+100)/2,-(shape.part(0).y+200))
 turtle.pendown()
 p=394
 d17=shape.part(17).x-shape.part(1).x
 print("range",d)
 davg=37
 ratio=(davg/d17)
 print("ratio",ratio)
 for b in range(0,67):
 print(b,shape.part(b).x,shape.part(b).y);
 if shape.part(b).x>xlimit1:
 xlimit1=shape.part(b).x
 if shape.part(b).y>ylimit1:
 ylimit1=shape.part(b).y
 if shape.part(b).x<xlimit2:
 xlimit2=shape.part(b).x
 if shape.part(b).y<ylimit2:
 ylimit2=shape.part(b).y
 print("xlimit1",xlimit1)
 print("ylimit1",ylimit1)
 print("xlimit2",xlimit2)
 print("ylimit2",ylimit2)
 turtle.penup()
 turtle.setpos(((-shape.part(0).x+100)/2),((-shape.part(0).y+200)/2))
 turtle.pendown()
 turtle.speed(1)
 d16a=(xvalues[16]-xvalues[2])/ratio
 d15a=(xvalues[15]-xvalues[3])/ratio
 [AUTOMATIC CARICATURE GENERATION]
46
 d14a=(xvalues[14]-xvalues[4])/ratio
 d13a=(xvalues[13]-xvalues[5])/ratio
 d12a=(xvalues[12]-xvalues[6])/ratio
 d11a=(xvalues[11]-xvalues[7])/ratio
 d10a=(xvalues[10]-xvalues[8])/ratio
 d16=shape.part(16).x-shape.part(2).x
 d15=shape.part(15).x-shape.part(3).x
 d14=shape.part(14).x-shape.part(4).x
 d13=shape.part(13).x-shape.part(5).x
 d12=shape.part(12).x-shape.part(6).x
 d11=shape.part(11).x-shape.part(7).x
 d10=shape.part(10).x-shape.part(8).x
 for b in range(0,67):
 #time.sleep(1)
 print(b+shape.part(b).x,shape.part(b).y)
 if b==17:
 turtle.penup()
 turtle.setpos(((-shape.part(b+1).x+100)/2),((-shape.part(b+1).y+200)/2))
 turtle.pendown()
 if b==22:
 turtle.penup()

 #turtle.setpos((-shape.part(23).x+100)/2,-(shape.part(23).y+200)/2)

 turtle.setpos(((-shape.part(b+1).x+100)/2),((-shape.part(b+1).y+200)/2))
 turtle.pendown()
 if b==27:
 turtle.penup()
 [AUTOMATIC CARICATURE GENERATION]
47
 #turtle.setpos((-shape.part(28).x+100)/2,-(shape.part(28).y+200)/2)

 turtle.setpos(((-shape.part(b+1).x+100)/2),((-shape.part(b+1).y+200)/2))
 turtle.pendown()
 if b==36:
 turtle.penup()
 # turtle.setpos((-shape.part(37).x+100)/2,-(shape.part(37).y+200)/2)

 turtle.setpos(((-shape.part(b+1).x+100)/2),((-shape.part(b+1).y+200)/2))
 turtle.pendown()
 if b==42:
 turtle.penup()

 turtle.setpos(((-shape.part(b+1).x+100)/2),((-shape.part(b+1).y+200)/2))
 #turtle.setpos((-shape.part(43).x+100)/2,-(shape.part(43).y+200)/2)
 turtle.pendown()
 if b==48:
 turtle.penup()

 turtle.setpos(((-shape.part(b+1).x+100)/2),((-shape.part(b+1).y+200)/2))
 #turtle.setpos((-shape.part(49).x+100)/2,-(shape.part(49).y+200)/2)
 turtle.pendown()
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
 [AUTOMATIC CARICATURE GENERATION]
48
#include <iostream>
using namespace dlib;
using namespace std;
// ----------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
 try
 {
 if (argc == 1)
 {
 cout << "Give some image files as arguments to this program." << endl;
 return 0;
 }
 frontal_face_detector detector = get_frontal_face_detector();
 image_window win;
 // Loop over all the images provided on the command line.
 for (int i = 1; i < argc; ++i)
 {
 cout << "processing image " << argv[i] << endl;
 [AUTOMATIC CARICATURE GENERATION]
49
 array2d<unsigned char> img;
 load_image(img, argv[i]);
 // Make the image bigger by a factor of two. This is useful since
 // the face detector looks for faces that are about 80 by 
 [AUTOMATIC CARICATURE GENERATION]
50
 cout << "Hit enter to process the next image..." << endl;
 cin.get();
 }
 }
 catch (exception& e)
 {
 cout << "\nexception thrown!" << endl;
 cout << e.what() << endl;
 }
}
// ---------------------------------------------------------------------------------------

 x1=0;
 y1=0;
 #shape.part(b).y =shape.part(b).y+10

 x1=((-shape.part(b).x+100)/2)
 y1=((-shape.part(b).y+200)/2)
 if b>1 and b<17:
 if b==2 or b==16:
 if d16a<d16:
 if b==16:
 x1=((-shape.part(b).x+80)/2)
 if b==2:
 [AUTOMATIC CARICATURE GENERATION]
51
 x1=((-shape.part(b).x+120)/2)
 if d16a>d16:
 if b==16:
 x1=((-shape.part(b).x+90)/2)
 if b==2:
 x1=((-shape.part(b).x+110)/2)
 if b>2 and b<16:
 if b==3 or b==15:
 if d15a<d15:
 if b==15:
 x1=((-shape.part(b).x+80)/2)
 if b==3:
 x1=((-shape.part(b).x+120)/2)
 if d15a>d15:
 if b==15:
 x1=((-shape.part(b).x+90)/2)
 if b==3:
 x1=((-shape.part(b).x+110)/2)

 if b>3 and b<15:
 if b==4 or b==14:
 if d14a<d14:
 if b==14:
 x1=((-shape.part(b).x+80)/2)
 if b==4:
 x1=((-shape.part(b).x+120)/2)
 if d14a>d14:
 if b==14:
 x1=((-shape.part(b).x+90)/2)
 if b==4:
 x1=((-shape.part(b).x+110)/2)
 if b>4 and b<14:
 [AUTOMATIC CARICATURE GENERATION]
52
 if b==5 or b==13:
 if d13a<d13:
 if b==13:
 x1=((-shape.part(b).x+80)/2)
 if b==5:
 x1=((-shape.part(b).x+120)/2)
 if d13a>d13:
 if b==13:
 x1=((-shape.part(b).x+90)/2)
 if b==5:
 x1=((-shape.part(b).x+110)/2)
 if b>5 and b<13:
 if b==6 or b==12:
 if d12a<d12:
 if b==12:
 x1=((-shape.part(b).x+80)/2)
 if b==6:
 x1=((-shape.part(b).x+120)/2)
 if d12a>d12:
 if b==12:
 x1=((-shape.part(b).x+90)/2)
 if b==6:
 x1=((-shape.part(b).x+110)/2)
 if b>6 and b<14:
 if b==7 or b==11:
 if d11a<d11:
 if b==11:
 x1=((-shape.part(b).x+80)/2
 [AUTOMATIC CARICATURE GENERATION]
53
 x1=((-shape.part(b).x+90)/2)
 if b==7:
 x1=((-shape.part(b).x+110)/2)
 if b>7 and b<15:
 if b==8 or b==10:
 if d10a<d10:
 if b==10:
 x1=((-shape.part(b).x+80)/2)
 if b==8:
 x1=((-shape.part(b).x+120)/2)
 if d10a>d10:
 if b==10:
 x1=((-shape.part(b).x+90)/2)
 if b==8:
 x1=((-shape.part(b).x+110)/2)
 turtle.goto(x1,y1)

 #time.sleep(1)


 turtle.penup()

 turtle.setpos(((-shape.part(37).x+100)/2),((-shape.part(37).y+190)/2))
 turtle.pendown()
 turtle.circle(2)
 [AUTOMATIC CARICATURE GENERATION]
54
 turtle.penup()

 turtle.setpos(((-shape.part(43).x+100)/2),((-shape.part(43).y+190)/2))
 turtle.pendown()
 turtle.circle(2)
 turtle.penup()

 height, width =threshold.shape
 #print( height, width)
 turtle.speed(0)
 turtle.width(0)

 turtle.setpos(((-shape.part(6).x+100)/2),((-shape.part(6).y+195)/2))
 turtle.pendown()
 turtle.goto(((-shape.part(6).x+100)/2),(((-shape.part(6).y+195)/2)-30))
 turtle.penup()

 turtle.setpos(((-shape.part(9).x+100)/2),((-shape.part(8).y+195)/2))
 turtle.pendown()
 turtle.goto(((-shape.part(9).x+100)/2),(((-shape.part(8).y+195)/2)-30))
 turtle.penup()

 for n in range(int((ylimit2-100)/2),int((ylimit1-40)),4):
 turtle.tracer(0,0)
 for x in range(int((xlimit2-100)/2),int(xlimit1+100/2),4):
 if x>xlimit1+50:
 continue
 if x<xlimit2-50:
 continue
 if threshold[n][x]!=0:
 #print(threshold[x][n])
 [AUTOMATIC CARICATURE GENERATION]
55
 turtle.setposition((-x+106)/2,(-n+202)/2)
 turtle.pendown()
 turtle.dot()
 turtle.penup()
 #turtle.tracer(0, 0)

 # Draw the face landmarks on the screen.
 turtle.update()
 #turtle.mainloop()
 turtle.pendown()
 turtle.setpos(((-shape.part(37).x+100)/2),((-shape.part(37).y+195)/2))
 turtle.goto(((-shape.part(37).x+100)/2),(((-shape.part(37).y+195)/2)+10))

 dlib.hit_enter_to_continue()