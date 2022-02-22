from flask import Flask, request, render_template
import cv2
from cv2 import FONT_HERSHEY_PLAIN
import numpy as np
from PIL import Image
import os
import string
import random

app = Flask(__name__)

app.config['INITIAL_FILE_UPLOADS'] = 'static/uploads'

@app.route("/", methods=["GET", "POST"])
def index():
    
    if request.method == "GET":
        full_filename = 'assets/white_bg.png'
        return render_template("index.html", full_filename = full_filename)

    if request.method == "POST":

        file_uploaded = request.files['file_upload']
        uploded_filename = file_uploaded.filename

        letters = string.ascii_lowercase
        fname = ''.join(random.choice(letters) for i in range(10)) + '.png'

        simg = Image.open(file_uploaded)
        simg.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], fname))

        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        img = cv2.imread("static/uploads/" + fname)
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_DUPLEX
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(img, (x , y - 35), (x + w , y + h), (0, 0, 0), 2)
                cv2.rectangle(img, (x, y), (x + w, y - 35), (0, 0, 0), -2)
                cv2.putText(img, label, (x, y - 10), font , 1, (255, 255, 255), 1)

        cv2.imwrite('static/uploads/' + fname, img)

        rname = 'uploads/' + fname

        return render_template("index.html", full_filename = rname)


if __name__ == "__main__":
    app.run(debug=True)