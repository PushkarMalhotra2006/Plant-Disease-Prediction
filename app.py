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
weights_file = r"AIML\DL Projects\Plant-Disease-Prediction\Plant_Disease_MobileNet_model_weights.weights.h5"

try:
    model = load_final_fixed_model(weights_file)
    print("\n✅ SUCCESS: Model loaded and compiled!")
    print("You can now proceed to model.predict()")
except Exception as e:
    print(f"\n❌ Error occured")

import numpy as np
from PIL import Image
import json

def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    return img

def predict_image(img):
  img = preprocess_image(img)

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

def display_prediction(img):
    class_name = predict_image(img)

    plant,disease,info = clean_output(class_name)

    return f"""
    <div style="font-size:18px; line-height:1.6; padding:10px;">
        <h1 style = "text-align:left; margin-bottom=10px">Prediction Output 👇</h1>
        <br>
        <h2>🌿 Plant: <b>{plant}</b></h2>
        <h2>🦠 Disease: <b>{disease}</b></h2>
        <p><b>📖 Information:</b>{info}</p>
    </div>
    """

with open("C:\Coding\AIML\DL Projects\Plant-Disease-Prediction\Plant_class_mappings.json", "r") as f:
    index_to_class = json.load(f)
with open("C:\Coding\AIML\DL Projects\Plant-Disease-Prediction\Disease_info.json","r") as d:
    disease_info = json.load(d)
print("file loaded")



#img = Image.open("C:\Coding\AIML\DL Projects\Plant-Disease-Prediction\grape leaf blight 2.webp")
#print(display_prediction(img))



import gradio as gr

with gr.Blocks(title="Plant Disease Detection",css="""
#output_box {
    min-width: 600px;
    font-size: 20px;
}
""") as app:

    gr.Markdown("""
<div style="text-align: center; width: 100%;">
<h1>🌿 Plant Disease Detection</h1>
<h4>Detect plant diseases using AI</h4>

<p style = "font-size:18px;"><b>Plants Covered:</b> Apple • Blueberry • Cherry • Corn • Grape • Orange • Peach • Pepper • Potato • Raspberry • Soybean • Squash • Strawberry • Tomato</p>

<p style = "font-size:16px;"><b>Instruction:</b> Upload clean image of the leaf</p>

<p style = "font-size:17px;"><b>Note:</b> Model is not generalized properly due to lack of data</p>
                
<hr>
</div>
""", container=True)
    
    gr.Markdown("Upload below to detect disease and get information.")

    with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Leaf Image")
            output_text = gr.HTML(elem_id="output_box")

    submit_btn = gr.Button("🔍 Analyze")

    submit_btn.click(
        fn=display_prediction,
        inputs=image_input,
        outputs=output_text,
        show_progress="full"
    )

    gr.Markdown("""
    ---
    
    <div style="text-align: center; font-size: 14px; color: gray;">
    ⚠️ Model can predict wrong answers due to lack of data.
    </div>
    """)

app.launch()