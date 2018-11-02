#!/usr/bin/env python
from flask import Flask, render_template, Response, request
import cv2
import sys
import numpy
import datetime
import time
import os 

found = None
# The amount of time to wait when a face is found before taking a picture
wait_time = 5

# OpenCV gets the face but FaceNet needs the whole head
image_padding = 30
image_x = 640
image_y = 480

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'pictures')
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

# Face classifier from OpenCV
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#face_model = load_model('my_model.h5', custom_objects={'triplet_loss': triplet_loss})

# Make the flask app
app = Flask(__name__)

# The starting index of our webpage
@app.route('/')
def index():
    return render_template('index.html', message = "Video Streaming Demonstration")
'''
# I do not think we need this code that is why  commented this out
# If we need it feel free to uncomment will remove if not
def gen():
    i=1
    while i<10:
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+str(i)+b'\r\n')
        i+=1'''

# Function for getting the frame from webcam using opencv
def get_frame():
    timeFoundFace = 0
    timeForNextStep = time.time() + wait_time
    x1, y1, x2, y2 = 0, 0, 0, 0
    finishedProcessing = False

    # Make the video catpture
    camera=cv2.VideoCapture(0)
    template = 'picture'
    # Constant loop for getting the image
    while True:
        found_Face = False
    	# get an image from the video capture
        retval, im = camera.read()
        im = cv2.resize(im, (image_x, image_y))
        # change the image into a grey image
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # get an array of face locations that is returned from the cascade
        faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

        # for every face we have we draw a rectangle around it
        for (x, y, w, h) in faces:
            found_Face = True
            x1 = x - image_padding
            y1 = y - image_padding
            x2 = x + w + image_padding
            y2 = y + h + image_padding

            text_x = x1
            text_y = y1 - 10
            cv2.rectangle(im, (x1,y1), (x2, y2), (255,0,0), 1)
            break

        # Get the text to print on the image
        if found_Face:
            timeFoundFace = time.time()
            stroke = 1
            if timeFoundFace >= timeForNextStep:
                if finishedProcessing:
                    text = "Done"
                else:
                    text =  "Processing"
                    save_Picture(im[y1+2:y2-2, x1+2:x2-2])
                    finishedProcessing = True
            else:
                time_left = int(timeForNextStep - timeFoundFace)
                if time_left == 0:
                    text = "Taking Pic"
                else:
                    text = "Waiting for " + str(time_left) + " sec"
        else:
            timeFoundFace = 0
            timeForNextStep = time.time() + wait_time
            text_x = 25
            text_y = 35
            stroke = 2
            finishedProcessing = False
            text =  "No Face Found"

        # Write the time to the image wil use this as a way to write instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0,0,0)
        location = (text_x, text_y)
        cv2.putText(im, text, location, font, 1, color, stroke, cv2.LINE_AA)


        imgencode=cv2.imencode('.jpg',im)[1]
        # then convert it to a string of values
        stringData=imgencode.tostring()
        # return the value. yield means it will return the value but keep running the code
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

    # when the loops breaks delete the camera
    del(camera)

@app.route('/calc')
def calc():
	# get a response for the img in the index.html
	# get_frame will constantly return a string of values
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/found')
def found():
       return render_template('Found.html')
@app.route('/not_found')
def not_found():
       return render_template('Not_Found.html')

if found: 
   def itIsFound():
       return redirect(url_for('found'))
else: 
   def itIsNotFound():
       return redirect(url_for('not_found'))
	

def save_Picture(image):
    filename = "test.jpg"
    cv2.imwrite(os.path.join(final_directory , filename), img=image)

if __name__ == '__main__':
	app.run(host='localhost', debug=True, threaded=True)
