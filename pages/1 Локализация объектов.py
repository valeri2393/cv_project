import streamlit as st
import torch
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Streamlit title
st.title('Локализация объектов с помощью модели ResNet18')

@st.cache_resource
def get_model():
    class LocModel(nn.Module):
        def __init__(self):
            super().__init__()

            # Load the pretrained ResNet18 model
            self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

            # Classifier layer
            self.clf = nn.Sequential(
                nn.Linear(512 * 8 * 8, 128),
                nn.Sigmoid(),
                nn.Linear(128, 3)
            )

            # Bounding box regression layer
            self.box = nn.Sequential(
                nn.Linear(512 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                nn.Sigmoid()
            )

        def forward(self, img):
            resnet_out = self.feature_extractor(img)
            resnet_out = resnet_out.view(resnet_out.size(0), -1)
            pred_classes = self.clf(resnet_out)
            pred_boxes = self.box(resnet_out)
            return pred_classes, pred_boxes

    # Load the model and weights
    model = LocModel()
    model.load_state_dict(torch.load('models/model_weights.pth'))
    model.eval()
    return model

# Get the model
model = get_model()

def predict_image(image):
    transform = T.Compose([
        T.Resize((227, 227)),
        T.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        pred_classes, pred_boxes = model(input_tensor)

    _, predicted_class = torch.max(pred_classes, 1)
    predicted_class = predicted_class.item()
    predicted_box = pred_boxes.squeeze().cpu().numpy()

    image_width, image_height = image.size
    xmin = int(predicted_box[0] * image_width)
    ymin = int(predicted_box[1] * image_height)
    xmax = int(predicted_box[2] * image_width)
    ymax = int(predicted_box[3] * image_height)

    return xmin, ymin, xmax, ymax, predicted_class

# File uploader
uploaded_file = st.file_uploader('Upload an image', type=['jpeg', 'jpg', 'png'])
results = None
lcol, rcol = st.columns(2)

# Image and prediction
if uploaded_file:
    image = Image.open(uploaded_file)
    xmin, ymin, xmax, ymax, predicted_class = predict_image(image)
    lcol.image(image, caption='Uploaded Image.', use_column_width=True)

    fig, ax = plt.subplots()
    ax.imshow(image)

    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    class_names = ['cucumber', 'eggplant', 'mushroom']
    predicted_label = class_names[predicted_class]
    plt.text(xmin, ymin, predicted_label, bbox={'facecolor': 'white', 'pad': 2})

    results = fig

# Display the results
if results:
    rcol.pyplot(results)
