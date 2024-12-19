from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import io

app = Flask(__name__)

# Load your models
standard_model = load_model("skin_lesion_model.keras")
mc_model = load_model("monte_carlo_model.keras")

# Define image size
img_size = 128

# Define the class mapping
class_mapping = {
    0: "nv",
    1: "mel",
    2: "bkl",
    3: "bcc",
    4: "akiec",
    5: "df",
    6: "vasc"
}

def preprocess_image(image):
    image = image.resize((img_size, img_size))
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def monte_carlo_predictions(model, input_data, n_simulations=50):
    preds = []
    for _ in range(n_simulations):
        preds.append(model(input_data, training=True))
    preds = np.stack(preds, axis=0)
    mean_pred = np.mean(preds, axis=0)
    std_pred = np.std(preds, axis=0)
    return mean_pred, std_pred

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = load_img(io.BytesIO(file.read()), target_size=(img_size, img_size))
    input_data = preprocess_image(image)

    # Standard prediction
    standard_pred = standard_model.predict(input_data)
    predicted_class = np.argmax(standard_pred, axis=-1)[0]

    # Monte Carlo prediction
    mean_pred, uncertainty = monte_carlo_predictions(mc_model, input_data)

    # Map the predicted class index to the label
    predicted_label = class_mapping[predicted_class]

    return jsonify({
        'predicted_class': predicted_label,
        'mean_prediction': mean_pred.tolist(),
        'uncertainty': uncertainty.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)