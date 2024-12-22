import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from streamlit_drawable_canvas import st_canvas
import cv2
import string

# --- Model Definition (Adapted from the repository) ---
class LineModelCtc(torch.nn.Module):
    def __init__(self, num_chars, cnn_out_size=512, lstm_hidden_size=256, lstm_layers=2, dropout=0.5):
        super().__init__()
        self.num_chars = num_chars
        self.cnn_out_size = cnn_out_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        # CNN
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=256, out_channels=cnn_out_size, kernel_size=3, padding=1)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.bn5 = torch.nn.BatchNorm2d(cnn_out_size)

        # LSTM
        self.lstm = torch.nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=True,
        )

        # Linear
        self.fc = torch.nn.Linear(lstm_hidden_size * 2, num_chars)

    def forward(self, x):
        # CNN
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.max_pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.max_pool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.max_pool(x)
        x = self.relu(self.bn5(self.conv5(x)))

        # Rearrange for LSTM
        x = x.permute(3, 0, 2, 1)  # (W, B, H, C)
        x = x.view(x.shape[0], x.shape[1], -1)  # (W, B, H * C)

        # LSTM
        x, _ = self.lstm(x)

        # Linear
        x = self.fc(x)

        return x

# --- Character Set ---
# Get all printable characters, including digits, punctuation, and whitespace
all_printable_chars = string.printable
characters = sorted(list(set(list(all_printable_chars))))
num_classes = len(characters)
char_to_num = {char: i for i, char in enumerate(characters)}
num_to_char = {i: char for i, char in enumerate(characters)}

# --- Model Loading ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LineModelCtc(num_chars=num_classes + 1).to(device) # +1 for CTC blank token
    model_state_dict = torch.load(
        "models/line_lstm_ctc_weights.h5",
        map_location=device
    )
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

model = load_model()

# --- Preprocessing Functions ---
def preprocess_image_for_prediction(image):
    """Preprocesses the drawn image for prediction."""
    image = ImageOps.grayscale(image)
    target_width, target_height = 128, 32
    image = image.resize((target_width, target_height))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = img_array.transpose((1, 0, 2))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_for_visualization(image):
    """Preprocesses the drawn image for visualization steps."""
    image = ImageOps.grayscale(image)
    image = image.resize((256, 64))
    img_array = np.array(image)
    return img_array

# --- Visualization Functions ---
def visualize_segmentation(img_array):
    """Visualizes a basic segmentation process."""
    st.write("#### 1. Segmentation (Simplified Example)")
    _, thresh = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV)
    st.image(thresh, caption="Binarized Image", width=300)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.cvtColor(img_array.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    st.image(contour_img, caption="Contours (Potential Character Boundaries)", width=300)

def visualize_normalization(img_array):
    """Visualizes a simple normalization process."""
    st.write("#### 2. Normalization")
    equalized = cv2.equalizeHist(img_array)
    st.image(equalized, caption="Histogram Equalization", width=300)

def plot_conv_layer_output(layer_outputs, layer_num):
   """Visualizes feature maps from a convolutional layer."""
   num_filters = layer_outputs.shape[-1]
   cols = 8
   rows = (num_filters + cols - 1) // cols
   fig, axes = plt.subplots(rows, cols, figsize=(12, 2 * rows))
   axes = axes.ravel()
   for i in range(num_filters):
       axes[i].imshow(layer_outputs[0, :, :, i], cmap="viridis")
       axes[i].axis("off")
   for i in range(num_filters, len(axes)):
       axes[i].set_visible(False)
   st.pyplot(fig)

def plot_prediction(prediction_text, confidence_scores=None):
    """Displays the prediction text and optionally confidence scores."""
    st.write("### Prediction")
    st.write(f"Predicted Text: **{prediction_text}**")

# --- Prediction Function ---
def decode_prediction(pred):
    pred = torch.tensor(pred)
    pred = pred.permute(1, 0, 2)
    pred = F.softmax(pred, dim=2)
    output_text = []
    for i in range(pred.shape[0]):
        max_index = torch.argmax(pred[i, 0, :])
        if max_index != 0:
            output_text.append(characters[max_index.item() - 1])
        else:
            output_text.append("")
    final_text = ""
    for i, char in enumerate(output_text):
        if i == 0 or char != output_text[i - 1]:
            final_text += char
    return final_text

# --- Streamlit App ---
st.title("Handwritten Text Recognition with Visualization")

# --- Sidebar: Drawing Canvas ---
st.sidebar.header("Draw Text")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    width=512,
    height=128,
    drawing_mode="freedraw",
    key="canvas",
)

# --- Main Area: Visualization and Prediction ---
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype("uint8"))
    input_image_pred = preprocess_image_for_prediction(img)
    input_image_vis = preprocess_image_for_visualization(img)

    if st.sidebar.button("Submit"):
        # --- Visualization ---
        st.write("## Visualization Steps")

        st.write("### Input Image")
        st.image(img, width=300)

        visualize_segmentation(input_image_vis)
        visualize_normalization(input_image_vis)

        # --- Model Prediction ---
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_image_pred).float()
            device = next(model.parameters()).device
            input_tensor = input_tensor.to(device)
            intermediate_layer_model = torch.nn.Sequential(*list(model.children())[:5])
            layer_outputs = intermediate_layer_model(input_tensor)
            prediction = model(input_tensor)

        # --- Decode Prediction ---
        decoded_text = decode_prediction(prediction.cpu().numpy())

        # --- Visualize Feature Maps ---
        for i, layer_output in enumerate([layer_outputs]):
            st.write(f"### Layer (Convolutional)")
            plot_conv_layer_output(layer_output.cpu().numpy(), i)

        # --- Display Prediction ---
        st.write("### Predicted Text")
        st.write(decoded_text)
