from flask import render_template, request, jsonify, flash, abort,g
import numpy as np
import cv2
import base64
import random
from draw import app
import keras
import re
import os
import tensorflow as tf


classes = ['backpack', 'banana', 'baseball', 'bed', 'cake', 'cow', 'door', 'fish', 'flower', 'hot dog', 'rabbit',
           'sheep', 'The Great Wall of China']

graph = tf.get_default_graph()
model = keras.models.load_model(os.getcwd() + "/model.h5")


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/draw')
def draw():
    label = random.choice(classes)
    return render_template("draw.html", data=label)


@app.route('/play')
def play():
    label = random.choice(classes)
    return render_template("play.html", data=label)


@app.route('/get', methods=['post'])
def a():
    args = request.get_json()
    url = args['url']
    result = re.search("data:image/(?P<ext>.*?);base64,(?P<data>.*)", url, re.DOTALL)
    if result:
        ext = result.groupdict().get("ext")
        data = result.groupdict().get("data")

    else:
        return "图片为空"
    img_b64decode = base64.urlsafe_b64decode(data)
    img_array = np.fromstring(img_b64decode, np.uint8)  # 转换np序列
    img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)  # 转换Opencv格式
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    change_image_size = cv2.resize(imgray, (28, 28), interpolation=cv2.INTER_AREA)
    change_image_size = 255 - change_image_size
    with graph.as_default():
        y_pred = model.predict(change_image_size.reshape(1, 28, 28, 1))
        y = classes[np.argmax(y_pred)]
    return jsonify({'class': y })
