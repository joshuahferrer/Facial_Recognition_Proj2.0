#!/usr/bin/env python
from flask import Flask, render_template, Response, request
import cv2
import sys
import numpy
import datetime
# Make the classifier objecs

# This is the step number for when we take a users picture
takePicuteStep = 1

# OpenCV gets the face but FaceNet needs the whole head
image_padding = 25

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Make the flask app
app = Flask(__name__)
# Render template, honestly idk
@app.route('/')
def index():
    return render_template('index.html', message = "Video Streaming Demonstration")

# IDK
def gen():
    i=1
    while i<10:
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+str(i)+b'\r\n')
        i+=1

# Function for getting the frame from webcam using opencv
def get_frame():
    # Make the video catpture
    # The 0 is the index of the video camera default is 0
    camera=cv2.VideoCapture(0) #this makes a web cam object
    i=1
    num = 1
    template = 'picture'
    # Constant loop for getting the image
    while True:
    	# get an image from the video capture
        retval, im = camera.read()
        # change the image into a grey image
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # get an array of face locations that is returned from the cascade
        faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

        # for every face we have we draw a rectangle around it
        for (x, y, w, h) in faces:
            x1 = x - image_padding
            y1 = y - image_padding
            x2 = x + w + image_padding
            y2 = y + h + image_padding

            roi = im[y1:y2, x1:x2]  # new line
            if num <=5:
               strNum = str(num)
               nameOfFile = template + strNum + '.jpg'
               #time.sleep(4)
               cv2.imwrite(filename=nameOfFile, img=roi)
               num = num + 1
            cv2.rectangle(im, (x1,y1), (x2, y2), (255,0,0), 2)


        # Write the time to the image wil use this as a way to write instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0,0,0)
        stroke = 1
        text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(im, text, (25,25), font, 1, color, stroke, cv2.LINE_AA)


        imgencode=cv2.imencode('.jpg',im)[1]
        # then convert it to a string of values
        stringData=imgencode.tostring()
        # return the value. yield means it will return the value but keep running the code
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
        i+=1

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

if __name__ == '__main__':
	app.run(host='localhost', debug=True, threaded=True)








#if __name__ == '__main__':
        #main()
