from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ResNet18 Model with MC Dropout
def initialize_model(num_classes):
    model = models.resnet18(weights=None)  # Do not load default weights
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),  # Monte Carlo Dropout
        torch.nn.Linear(in_features, num_classes)
    )
    return model

# Load Model with Correct Structure
model = initialize_model(num_classes=7)
checkpoint = torch.load("best_model.pth", map_location=device)  # Load saved weights
model.load_state_dict(checkpoint)  # Load state_dict properly
model.to(device)
model.eval()  # Set to evaluation mode

# Enable Monte Carlo Dropout
def enable_dropout(model):
    """Enable dropout layers during inference for Monte Carlo Dropout."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

# Define Image Transform
img_size = 224  # Match model input size
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define Class Mapping
class_mapping = {
    0: "nv",
    1: "mel",
    2: "bkl",
    3: "bcc",
    4: "akiec",
    5: "df",
    6: "vasc"
}

# Monte Carlo Dropout Inference
def monte_carlo_predictions(model, input_tensor, n_samples=50):
    """Perform Monte Carlo Dropout with multiple stochastic forward passes."""
    enable_dropout(model)  # Activate dropout layers
    preds = []

    for _ in range(n_samples):
        with torch.no_grad():
            output = model(input_tensor)
        preds.append(F.softmax(output, dim=1).cpu().detach().numpy())

    preds = np.stack(preds)  # Shape: (n_samples, batch_size, num_classes)
    mean_pred = np.mean(preds, axis=0)  # Mean prediction
    std_pred = np.std(preds, axis=0)  # Uncertainty (variance)
    entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=1)  # Entropy

    return mean_pred, std_pred, entropy

# Compute OOD Threshold
def compute_ood_threshold():
    """Set OOD threshold based on dataset-wide uncertainty statistics."""
    uncertainty_distribution = [0.003, 0.007, 0.012, 0.020]  # Example dataset values
    return np.percentile(uncertainty_distribution, 95)  # Set threshold at 95th percentile

OOD_THRESHOLD = compute_ood_threshold()  # Dynamically set OOD threshold

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Standard Prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        standard_pred = F.softmax(output, dim=1).cpu().numpy()

    predicted_class = np.argmax(standard_pred, axis=-1)[0]
    predicted_label = class_mapping[predicted_class]

    # Monte Carlo Prediction
    mean_pred, uncertainty, entropy = monte_carlo_predictions(model, input_tensor)

    # **OOD Detection Based on Uncertainty**
    is_ood = entropy[0] > OOD_THRESHOLD  # Check if sample is OOD

    # Logging for debugging
    logging.info(f"Predicted Label: {predicted_label}")
    logging.info(f"Mean Prediction: {mean_pred}")
    logging.info(f"Uncertainty Scores: {uncertainty}")
    logging.info(f"Max Uncertainty: {max(uncertainty[0])}")
    logging.info(f"Entropy Score: {entropy[0]}")
    logging.info(f"OOD Detected: {is_ood}")

    return jsonify({
        'predicted_class': predicted_label,
        'mean_prediction': mean_pred.tolist(),
        'uncertainty': uncertainty.tolist(),
        'entropy': entropy.tolist(),
        'is_ood': bool(is_ood)  # OOD detection result
    })

if __name__ == '__main__':
    app.run(debug=True)
