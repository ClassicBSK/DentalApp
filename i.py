from flask import Flask, request, send_file
import torch
import torchvision
from torchvision.io import read_image
from torchvision.transforms import v2 as T
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import os
from PIL import Image
import numpy as np
from flasgger import Swagger

app = Flask(__name__)
swagger=Swagger(app)
# Load the trained Mask R-CNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'mask-rcnn.pt'  # Ensure this file is in the same directory
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found in {os.getcwd()}")
model = torch.load(model_path, map_location=device, weights_only=False)  # Set weights_only=False
model.eval()
model.to(device)

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
        # Get the uploaded image
        if 'image' not in request.files:
            return {'error': 'No image provided'}, 400
        file = request.files['image']
        image_path = 'temp_input.png'
        file.save(image_path)

        # Read and preprocess the image
        image = read_image(image_path)
        eval_transform = get_transform()
        x = eval_transform(image)

        # Convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)

        # Run inference
        with torch.no_grad():
            predictions = model([x])
            pred = predictions[0]

        # Process the output
        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        image = image[:3, ...]
        pred_labels = [f"object: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
        pred_boxes = pred["boxes"].long()
        output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

        masks = (pred["masks"] > 0.7).squeeze(1)
        output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

        # Save the output image
        output_path = 'output.png'
        plt.figure(figsize=(12, 12))
        plt.imshow(output_image.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        # Clean up temporary input file
        os.remove(image_path)

        # Return the processed image
        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)