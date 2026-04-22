# Plant-Disease-Prediction



\# 🌿 Plant Disease Detection using Deep Learning



An end-to-end deep learning application that detects plant diseases from leaf images using a MobileNetV2-based CNN model. The project includes model training, inference pipeline, and a fully deployed interactive web app.



\---



\## 🚀 Live Demo



👉 https://huggingface.co/spaces/PushkarCoder20/Plant-Disease-Detection



\---



\## 🧠 Project Overview



This project aims to identify plant diseases from leaf images using a lightweight deep learning model. It is designed as a complete pipeline from training to deployment.



The model predicts the disease class and provides additional information about the detected condition.



\---



\## ✨ Features



\- 🌿 Detects plant diseases from leaf images  

\- 🧠 Uses MobileNetV2 (Transfer Learning)  

\- ⚡ Fast inference with lightweight model  

\- 📊 Supports multiple plant categories  

\- 📖 Displays disease information alongside prediction  

\- 🌐 Deployed interactive UI using Gradio  

\- 🚀 Hosted on Hugging Face Spaces  



\---



\## 🪴 Supported Plants



\- Apple  

\- Blueberry  

\- Cherry  

\- Corn  

\- Grape  

\- Orange  

\- Peach  

\- Pepper  

\- Potato  

\- Raspberry  

\- Soybean  

\- Squash  

\- Strawberry  

\- Tomato  



---



\## 🏗️ Tech Stack



\- Python  

\- TensorFlow / Keras  

\- MobileNetV2  

\- NumPy  

\- Pillow  

\- Gradio  

\- Hugging Face Spaces  



---



\## 📂 Project Structure


Plant-Disease-Detection/

│

├── app.py                                             # Gradio application

├── requirements.txt                                   # Dependencies

├── README.md

│

├── Plant\_Disease\_MobileNet\_model\_weights.weights.h5   # Trained model weights

├── Plant\_class\_mappings.json                          # Class index → label mapping

├── Disease\_info.json                                  # Disease descriptions

│

├── Plant\_Disease\_Detection.ipynb                      # Training notebook

│

├── Other models/                                      # Additional/experimental models

│

├── .gitignore

└── .gitattributes

\```



---



\## ⚙️ How It Works



1\. User uploads a leaf image  

2\. Image is resized and preprocessed  

3\. Model predicts disease class  

4\. Class index is mapped to label  

5\. Disease information is displayed  



---



\## 🧪 Model Details



\- Architecture: MobileNetV2 (Transfer Learning)  

\- Input Size: 224 × 224  

\- Output: 38 classes  

\- Framework: TensorFlow / Keras  



---



\## ⚠️ Limitations



\- Model is trained on a limited dataset  

\- May not generalize well to real-world images  

\- Accuracy may vary for unseen conditions  



---



\## 📸 Example Output



\- 🌿 Plant: Tomato  

\- 🦠 Disease: Early Blight  

\- 📖 Info: Causes concentric rings on leaves  



---



\## 🚀 Deployment



The application is deployed using:



👉 Hugging Face Spaces (Gradio)



---



\## 👨‍💻 Author



PushkarMalhotra2006





