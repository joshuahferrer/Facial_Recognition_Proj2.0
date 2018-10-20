#!/usr/bin/env python
from flask import Flask, render_template, Response, request
import cv2
import sys
import numpy
import time

#import tkinter.messagebox



# from flask_sqlalchemy import SQLAlchemy #database
# app = Flask(__name__)  #database
#
# app.config['SQLAlCHEMY_DATABASE_URI'] = 'mysql://scott:tiger@localhost/mydatabase' #database
# #
# db = SQLAlchemy(app)
#
# class Example(db.Model):
#       __tablename__ = 'example'
#       id = db.Column('id',db.Integer,primary_key=True)
#       data = db.Column('data',db.Unicode)

#
# class User(db.Model):
#     id = db.Column(db.Integer,primary_key=True)
#     username = db.Column(db.String(20),unique=True, nullable=False)
#     email = db.Column(db.String(20),unique=True, nullable=False)
#     image_file = db.Column(db.String(20), ullable=False, default='default.jpg')
#     password=  db.Column(db.String(60), ullable=False)
#
#     def __repr__(self):
#         return f"User('{self.username}','{self.email}','{self.image_file}')"

# Make the classifier objecs

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
nose_cascade = cv2.CascadeClassifier("Nariz.xml")

# Make the flask app
app = Flask(__name__)
# Render template, honestly idk
@app.route('/')
def index():
    return render_template('index.html')

# IDK
def gen():
    i=1
    while i<10:
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+str(i)+b'\r\n')
        i+=1

# Function for getting the frame from webcam using opencv
def get_frame():

    camera_port=0

    ramp_frames=100
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
        #num = 1
        #template = 'picture'
        #@app.route("/", methods=['GET', 'POST'])
        for (x, y, w, h) in faces:
                roi = im[y:y+h, x:x+w]  # new line
                if num <=5:
                       strNum = str(num)
                       nameOfFile = template + strNum + '.jpg'
                       #time.sleep(4)
                       cv2.imwrite(filename=nameOfFile, img=roi)
                       num = num + 1
                       #if request.method == 'POST'
                       #if request.form['submit_button'] == 'Do Something':
                       #@app.route("/", methods=['GET', 'POST'])
                       #def index():
                        #   if request.method == 'POST':
                        #        if request.form.get('button') == 'pic':
                        #cv2.imwrite(filename=nameOfFile, img=roi)
                        #num = num + 1
                         #  elif request.method == 'GET':
                               # return render_template("index.html")
                        #           print("No Post Back Call")
                        #           return render_template("index.html")


            #cv2.imwrite("roi.jpg", roi)# new line
                cv2.rectangle(im, (x,y), (x+w, y+h), (255,0,0), 2)
        # I think this converts the image to a jpg
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
