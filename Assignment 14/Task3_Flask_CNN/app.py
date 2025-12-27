
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model.h5")
classes = ["person1", "person2", "person3"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None
    if request.method == "POST":
        file = request.files["image"]
        img_path = os.path.join("static", file.filename)
        file.save(img_path)

        img = image.load_img(img_path, target_size=(128,128))
        img = image.img_to_array(img)/255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        prediction = classes[np.argmax(pred)]

    return render_template("index.html", prediction=prediction, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
