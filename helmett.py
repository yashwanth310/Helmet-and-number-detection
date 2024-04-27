import streamlit as st
import cv2
import numpy as np
import tempfile
import imutils
from tensorflow.keras.models import load_model

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"hn.jpg"};base64,{encoded_string.decode()});
        background-size: cover
    
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('hn.jpg') 

# Set up OpenCV DNN model and load the helmet detection model
net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# Define the path to the YOLO .cfg file
cfg_file = 'yolov3-custom.cfg'

# Read the YOLO .cfg file and extract the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Output layer names should be populated with the appropriate layer names

# Load the helmet classification model
model = load_model('helmet-nonhelmet_cnn.h5')
st.write('Model loaded!!!')

# Function to detect helmet
def helmet_or_nohelmet(helmet_roi):
    try:
        helmet_roi = cv2.resize(helmet_roi, (224, 224))
        helmet_roi = np.array(helmet_roi, dtype='float32')
        helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
        helmet_roi = helmet_roi / 255.0
        return int(model.predict(helmet_roi)[0][0])
    except:
        pass

# Function to process image and detect helmet and number plate
def process_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), -1)
    img = imutils.resize(img, height=500)
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    COLORS = [(0, 255, 0), (0, 0, 255)]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                color = [int(c) for c in COLORS[class_id]]
                if class_id == 0:  # helmet
                    helmet_roi = img[max(0, y):max(0, y) + max(0, h) // 4, max(0, x):max(0, x) + max(0, w)]
                    c = helmet_or_nohelmet(helmet_roi)
                    cv2.putText(img, ['helmet', 'no-helmet'][c], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:  # number plate
                    x_h = x - 60
                    y_h = y - 350
                    w_h = w + 100
                    h_h = h + 100
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)
                    if y_h > 0 and x_h > 0:
                        h_r = img[y_h:y_h + h_h, x_h:x_h + w_h]
                        c = helmet_or_nohelmet(h_r)
                        cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h), (255, 0, 0), 10)
                        cv2.putText(img, ['helmet', 'no-helmet'][c], (x_h, y_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img

st.title("")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("Image uploaded!")
    if st.button("Process and Show"):
        with st.spinner("Processing image..."):
            result_img = process_image(uploaded_file)
            st.image(result_img, channels="BGR", caption="Processed Image")
