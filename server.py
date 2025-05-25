from flask import Flask, request, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import v2 as T
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks
import os
from PIL import Image
import torch.nn.functional as F
from flasgger import Swagger
import timm
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline

app = Flask(__name__)
swagger = Swagger(app)
CORS(app)  # Enable CORS for Flutter app

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Expand1000toImage(nn.Module):
    def __init__(self):
        super(Expand1000toImage, self).__init__()
        
        self.fc = nn.Linear(1000, 512 * 7 * 7)  # Fully connected layer
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     
        
        )
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 7, 7) 
        x = self.deconv_layers(x)
        return x
# Initialize TIMM EfficientViT-B0 model
class Hybrid_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet_fe=models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.converter=Expand1000toImage()
        self.vit_classifier = timm.create_model('vit_base_patch16_224_miil.in21k_ft_in1k', num_classes=6)

    def forward(self,x):
        x=self.resnet_fe(x) 
        x=self.converter(x)
        x=self.vit_classifier(x)
        return x
try:
    model=Hybrid_model()
    model_state=torch.load('llmmodels\\best_hybrid_model99.pth')
    model.load_state_dict(model_state)
    model.eval()
    model.to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Define data transformations to match separate script
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_transform():
    transforms = [T.ToDtype(torch.float, scale=True), T.ToPureTensor()]
    return T.Compose(transforms)

@app.route('/mask_rcnn', methods=['POST'])
def mask_rcnn():
    """
    Mask R-CNN Inference Endpoint
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: The image to process with Mask R-CNN
    responses:
      200:
        description: Returns an image with bounding boxes and masks
        content:
          image/png:
            schema:
              type: string
              format: binary
      400:
        description: No image provided
      500:
        description: Server error
    """
    try:
        mask_rcnn_path = r"llmmodels\\mask-rcnn.pt"
        if not os.path.exists(mask_rcnn_path):
            raise FileNotFoundError(f"Mask R-CNN model file {mask_rcnn_path} not found in {os.getcwd()}")
        mask_rcnn_model = torch.load(mask_rcnn_path, map_location=device, weights_only=False)
        mask_rcnn_model.eval()
        mask_rcnn_model.to(device)

        if 'image' not in request.files:
            return {'error': 'No image provided'}, 400
        file = request.files['image']
        image_path = 'temp_input.png'
        file.save(image_path)

        # Read and preprocess image
        image = read_image(image_path)
        eval_transform = get_transform()
        x = eval_transform(image)
        x = x[:3, ...].to(device)

        # Run Mask R-CNN inference
        with torch.no_grad():
            predictions = mask_rcnn_model([x])
            pred = predictions[0]

        # Process output
        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        image = image[:3, ...]
        masks = (pred["masks"] > 0.7).squeeze(1)
        output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors="blue")

        # Save output
        output_path = 'mask_rcnn_output.png'
        # plt.figure(figsize=(12, 12))
        plt.imshow(output_image.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight')
        # plt.show()
        plt.close()

        # Clean up
        os.remove(image_path)

        return send_file(output_path, mimetype='image/png')
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/vit', methods=['POST'])
def vit():
    """
    Hybrid VIT Classifier Disease Detection Endpoint
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: The dental image to classify for disease detection
    responses:
      200:
        description: Returns the predicted disease
        content:
          text/plain:
            schema:
              type: string
              description: Predicted disease (Calculus, Caries, Gingivitis, Hypodontia, Tooth Discoloration, Ulcers)
      400:
        description: No image provided or invalid image format
      500:
        description: Server error
    """
    try:
        # Validate image
        if 'image' not in request.files:
            return {'error': 'No image provided'}, 400
        file = request.files['image']
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return {'error': 'Invalid image format. Use PNG or JPEG.'}, 400

        image_path = 'temp_input.png'
        file.save(image_path)

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        print(f"Image tensor shape: {image_tensor.shape}")  # Should be [1, 3, 224, 224]
        print(f"Image tensor min: {image_tensor.min().item()}, max: {image_tensor.max().item()}")  # Check normalization
        plt.imshow(image)
        # plt.show()
        # Run EfficientViT-B0 inference
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            print(f"Raw Model Output (Logits): {output}")
            print(f"Outputs shape: {output.shape}")  # Should be [1, 6]
            probabilities = F.softmax(output, dim=1)
            print(f"Probabilities shape: {probabilities.shape}")  # Should be [1, 6]
            print(f"Full probabilities: {probabilities.tolist()}")
            confidence, class_idx = torch.max(probabilities, dim=1)
            print(f"Confidence: {confidence}, Class_idx: {class_idx}")
            print(f"Confidence shape: {confidence.shape}, Class_idx shape: {class_idx.shape}")  # Should be [1]

            if class_idx.shape != torch.Size([1]):
                return {'error': f'Unexpected class_idx shape: {class_idx.shape}'}, 500

            confidence = confidence.item()
            class_idx = class_idx.item()

            # Optional: Enforce confidence threshold (uncomment to enable)
            # if confidence < 0.7:
            #     return {'error': f'Low confidence ({confidence:.2f}). Prediction may be unreliable.'}, 500

        # Define class names to match training
        if class_idx==2:
            class_idx=0
        elif class_idx>2:
            class_idx-=1
        class_names = ['Calculus','Data caries','hypodontia','Mouth Ulcer','Tooth Discoloration']
        if class_idx >= len(class_names):
            return {'error': f'Invalid class index: {class_idx}'}, 500
        predicted_disease = class_names[class_idx]
        text=f"The disease identified is {predicted_disease}. {predicted_disease} dental disease is"
        generator = pipeline("text-generation",max_new_tokens=200, model="llmmodels\\models--gpt2\\gpt2")
        response = generator(text,num_return_sequences=1)
        # print(response[0]['generated_text'])
        obj={'classification':predicted_disease,'confidence':confidence,'details':response[0]['generated_text']}
        # Clean up
        os.remove(image_path)

        return obj

    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app,use_debugger=True,use_reloader=True)