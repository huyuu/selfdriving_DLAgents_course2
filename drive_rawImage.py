#!/usr/bin/env python

import argparse
import base64
import json
import time
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import scipy.misc
from io import BytesIO
from flask import Flask, render_template
from PIL import Image
from PIL import ImageOps

from tensorflow import keras

# Fix error with Keras and TensorFlow
import tensorflow as tf
#tf.python.control_flow_ops = tf

#csv 2018/10/03
import csv
#csv 2018/10/03 finish

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

e_fst = 0.0
e_sec = 0.0
sameCount = 0


@sio.on('telemetry')
def telemetry(sid, data):
    #define deflection
    global e_fst
    global e_sec
    #csvlist Initialization
    csvlist = []

    steering_angle = float(data["steering_angle"]) / 25.0
    throttle = float(data["throttle"])
    speed = float(data["speed"]) / 30.5
    # subPara = np.array([steering_angle, throttle, speed], dtype="float32").reshape(1, -1)
    subPara = np.array([steering_angle], dtype="float32").reshape(1, -1)

    # The current image from the center camera of the car
    image_path = data["image"]
    image = np.asarray(Image.open(BytesIO(base64.b64decode(image_path))))
    # image = getCenterDeviationWithImage(image)
    # image = image.reshape(1, image.shape[0], image.shape[1], 1).astype(np.float32) / 255.0
    image = image.reshape(1, image.shape[0], image.shape[1], 3).astype(np.float32) / 255.0

    # print(image_array.shape)
    # print(subPara.shape)

    # action_prob = model.predict([image, subPara], batch_size=None)[0]
    # action = np.argmax(action_prob)

    # steering_angle_before = model.predict(image, batch_size=None)[0, 0]
    steering_angle_before = model.predict([image, subPara], batch_size=None)[0, 0]
    # steering_angle_before = max(-0.5, min(0.5, steering_angle_before))
    # throttle_before = 1.0# if speed <= 0.9 else 0.0
    # throttle_before = 0.5 if abs(steering_angle_before) <= 0.25 else -1.0
    throttle_before = 1.0
    global sameCount
    if np.sign(steering_angle_before) == np.sign(steering_angle):
        sameCount += 1
    else:
        sameCount = 0
    if sameCount >= 3:
        throttle_before = -1.0 if speed >= 0.4 else 0.5
    # if abs(steering_angle_before) <= 0.1:
    #     throttle_before = 0.8
    # else:
    #     if speed <= 0.7:
    #         throttle_before = 0.7
    #     else:
    #         throttle_before = 0.0
    print(steering_angle_before, throttle_before)
    # calculate action
    # steering_angle_before, throttle_before = [0.0, 0.0]
    # if action == 0: # turn left
    #     if steering_angle <= -1e-2:
    #         steering_angle_before, throttle_before = [steering_angle-0.2, 1.0]
    #     else:
    #         steering_angle_before, throttle_before = [-0.2, 1.0]
    # elif action == 1: # turn right
    #     if steering_angle >= 1e-2:
    #         steering_angle_before, throttle_before = [steering_angle+0.2, 1.0]
    #     else:
    #         steering_angle_before, throttle_before = [0.2, 1.0]
    # else: # go straight
    #     steering_angle_before, throttle_before = [0.0, 1.0]
    # else: # break
    #     steering_angle_before, throttle_before = [0.0, -0.5]

    # print('Steering Angle: %f, \t Throttle: %f' % (steering_angle, throttle))
    send_control(steering_angle_before, throttle_before)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    model = keras.models.load_model('./continuousModel_rawImage.h5')

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
