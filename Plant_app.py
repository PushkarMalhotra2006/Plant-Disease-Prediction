import keras
from keras import layers, models

def load_final_fixed_model(weights_path):
    # 1. Base Model (MobileNetV2)
    # The log shows 'input_shape': [None, 1280] before the Dense layer, 
    # which confirms standard MobileNetV2 (Alpha=1.0)
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )

    # 2. Rebuild the exact Sequential structure from your error log
    model = models.Sequential([
        layers.InputLayer(shape=(224, 224, 3)),
        # You mentioned you used 1/255 rescaling in Colab
        layers.Rescaling(1./255), 
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', name='dense'),
        layers.Dropout(0.5),
        layers.Dense(38, activation='softmax', name='dense_1')
    ])

    # 3. Load the weights
    # We use skip_mismatch=True so it ignores that pesky 'quantization_config'
    model.load_weights(weights_path, skip_mismatch=True)
    
    # Manually compile since load_weights doesn't do it
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# --- RUN IT ---
# Use the .weights.h5 file path here
weights_file = r"C:\Coding\AIML\DL Projects\Plant-Disease-Prediction\Plant_Disease_MobileNet_model_weights.weights.h5"

try:
    model = load_final_fixed_model(weights_file)
    print("\n✅ SUCCESS: Model loaded and compiled!")
    print("You can now proceed to model.predict()")
except Exception as e:
    print(f"\n❌ Final hurdle: {e}")

import gradio as gr
import numpy as np
from PIL import Image
import json

def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    return img

def predict_image(img_path):
  img = preprocess_image(img_path)

  pred = model.predict(img)
  pred_class = np.argmax(pred)

  class_name = index_to_class[str(pred_class)]

  return class_name

def get_disease_info(class_name):
    return disease_info.get(class_name, "No information available.")

def clean_output(class_name):
  plant,disease = class_name.split("___")

  disease = disease.replace("_"," ")
  plant = plant.replace("_","")
  plant = plant.replace(",","")

  if disease=="healthy":
    disease = "No disease"

  info = get_disease_info(class_name)

  return plant, disease, info

def display_prediction(img_path):
    class_name = predict_image(img_path)

    plant,disease,info = clean_output(class_name)

    return f"Plant: {plant}\nDisease: {disease}\nInfo: {info}"

with open("C:\Coding\AIML\DL Projects\Plant-Disease-Prediction\Plant_class_mappings.json", "r") as f:
    index_to_class = json.load(f)
with open("C:\Coding\AIML\DL Projects\Plant-Disease-Prediction\Disease_info.json","r") as d:
    disease_info = json.load(d)
print("file loaded")



img = Image.open("C:\Coding\AIML\DL Projects\Plant-Disease-Prediction\grape leaf blight 2.webp")
print(display_prediction(img))