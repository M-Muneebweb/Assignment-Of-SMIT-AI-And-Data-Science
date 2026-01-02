from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("face_recognition_model.h5")

labels = ['Person1', 'Person2', 'Person3']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    img_path = ""

    if request.method == 'POST':
        file = request.files['image']
        img_path = os.path.join('static', file.filename)
        file.save(img_path)

        img = image.load_img(img_path, target_size=(128,128))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        prediction = labels[np.argmax(pred)]

    return render_template('index.html',
                           prediction=prediction,
                           img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
