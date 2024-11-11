import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

# Class names (mapped to the classes in the model)
class_names = [
    'crescent_gap', 'inclusion', 'oil_spot', 'punching_hole',
    'rolled_in_scale', 'scratches', 'silk_spot', 'waist_folding',
    'water_spot', 'welding_line'
]

# Function to load the pre-trained model and setup
def setup_model(model_path='..\models\\final_model.pth'):
    """
    Load the VGG16 model, modify the final layer, and load saved weights.
    """
    model = models.vgg16(pretrained=False)  # Load VGG16 without pre-trained weights
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=len(class_names))  # Modify for 10 classes
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load the saved weights
    model.eval()  # Set the model to evaluation mode
    return model

# Function to preprocess the image and predict the class
def predict_image(model, image):
    """
    Process the image, run prediction, and return predicted class label and probability.
    """
    # Define image transformation (to resize, normalize, and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224 (VGG16 input size)
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with VGG16 mean/std
    ])
    
    # Apply the transformation to the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make the prediction
    with torch.no_grad():  # Turn off gradients for inference
        output = model(image_tensor)  # Forward pass
        probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        predicted_class = torch.argmax(probabilities, dim=1)  # Get the predicted class (highest score)
        predicted_class_label = class_names[predicted_class.item()]  # Get the class label
        predicted_probability = round(probabilities[0][predicted_class].item() * 100,2)  # Get the probability

    return predicted_class_label, predicted_probability
