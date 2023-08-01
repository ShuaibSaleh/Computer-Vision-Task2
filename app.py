from flask import Flask, render_template, request
from flask_cors import CORS
import os
import base64  # convert from string to  bits
import json
import cv2
import numpy as np
import time
import calendar
import image as img
import ActiveContour as actcont
from skimage import io
from skimage.color import rgb2gray
from skimage.segmentation import active_contour
import Hough as fn
import json
# -------------------
from ActiveContour import contour
import math
import cv2


app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

CORS(app)


@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("activecontour.html")


@app.route("/activecontour", methods=["GET", "POST"])
def activecontour():
    if request.method == "POST":
        image_data = base64.b64decode(
            request.form["image_data"].split(',')[1])

        img_path = img.saveImage(image_data, "contour_img")
        # img_binary = img.readImg(img_path)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # parameters
        alpha = 30
        beta = 20
        gamma = 7
        contour_points = 60

        snake = contour(image, alpha, beta, gamma, contour_points)
        for j in range(100):
            snake.fit_snake()
        snake.draw_contour()

        len = snake.contour_len()
        # print("perimeter:","",len)
        area = snake.contour_area()
        # print("area:","",area)

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)

        output_path = './static/images/output/output.jpg'

        # return json.dumps({1: f'test'})
        return json.dumps({1: f'<img src="{output_path}?t={time_stamp}" id="ApplyEdges" alt="" >', 2: f'<p class="btn btn-success">Contour Perimeter: {round(len, 2)}</p>', 3: f'<p class="btn btn-success">Contour Area: {round(area, 2)}</p>'})

    else:
        return render_template("activecontour.html")


@app.route("/h", methods=['GET', 'POST'])
def houghtransform():
    return render_template("houghtransform.html")


@app.route("/houghtransform", methods=['GET', 'POST'])
def hough():
    if request.method == 'POST':
        path_output = ''
        selectbox = request.form['houghtransformselectbox']
        image_data = base64.b64decode(request.form["path"].split(',')[1])
        img_path = img.saveImage(image_data, "input")
        if selectbox == 'Circle':
            path_output = fn.circlehough(img_path)
            resizeimage=cv2.imread(path_output)
            resizeimage=cv2.resize(resizeimage , (420 ,330))
            cv2.imwrite(path_output , resizeimage) 
        elif selectbox == 'Line':
            image = cv2.imread(img_path)
            imagecanny = cv2.Canny(cv2.imread(img_path, 0), 50, 150)
            accumulator, thetas, rhos = fn.hough_Accumulator_thetas_dist(imagecanny)

            path_output = fn.save_hough_Line(image, accumulator, thetas, rhos)
        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        return json.dumps({1: f'<img src="{path_output}?t={time_stamp}" id="houghtransform" alt="" >'})
    else:
        return render_template('houghtransform.html')


if __name__ == "__main__":
    app.run(port=7757, debug=True)
