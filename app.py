import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your model
model = tf.keras.models.load_model("models/final_image_classifier.keras")

class_names = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor",
    2: "No Tumor",
    3: "Pituitary Tumor"
}

def predict_image(image):
    image = image.convert("RGB").resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)[0]
    class_index = np.argmax(prediction)
    confidence = float(prediction[class_index] * 100)
    return f"{class_names[class_index]} ({confidence:.2f}%)"

# Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="Brain Tumor Detection",
    description="Upload an MRI image to detect brain tumor type"
)

# Launch with sharing enabled
iface.launch(share=True)


