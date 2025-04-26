from flask import Flask, render_template, request
from keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np
import os

from data import class_names, remedies

model = load_model("model/vgg16_model.h5")
# model = load_model("model/vgg16_model.h5")
# model = load_model("model/inceptionv3_finetuned_model.h5")

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "GET":
        return render_template("index.html")

    file = request.files["file"]
    file_extension = os.path.splitext(file.filename)[1]
    newfilename = "uploaded" + file_extension
    img_path = "static/images/uploads/" + newfilename
    file.save(img_path)

    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x /= 255.0
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)
    class_idx = np.argmax(predictions[0])
    confidence_score = predictions[0][class_idx] * 100

    ## confidence scores 
    # for i, pred in enumerate(predictions[0]):
    #     class_label = class_names[i]
    #     confidence = pred * 100
    #     print(f"{class_label}: {confidence:.2f}%")

    print("confidence: ", confidence_score)
    confidence = "{:.2f}".format(confidence_score)

    if class_idx == 14 or class_idx == 4 or class_idx == 1:
        predicted_disease = "No disease detected (Healthy Leaf)"
        treatment = []
    else:
        predicted_disease = class_names[class_idx]
        treatment = remedies[class_idx]

    return render_template(
        "result.html",
        confidence=confidence,
        filename=newfilename,
        predicted_disease=predicted_disease,
        treatment=treatment,
    )


@app.errorhandler(Exception)
def handle_exception(error):
    print(error)
    return render_template("error.html", error=error), 500


if __name__ == "__main__":
    app.run()
