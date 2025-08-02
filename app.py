import joblib
import librosa
import numpy as np
import gradio as gr

# Load model
model = joblib.load("tree_sound_model.pkl")

# Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Prediction
def predict_tree(audio):
    if audio is None:
        return "Please upload an audio file."
    features = extract_features(audio).reshape(1, -1)
    prediction = model.predict(features)[0]
    return f"Predicted Tree Species: {prediction}"

# Gradio UI
app = gr.Interface(fn=predict_tree,
                   inputs=gr.Audio(type="filepath"),
                   outputs="text",
                   title="🌳 Tree Sound Classifier",
                   description="Upload a tree sound (.wav) to identify the species")

app.launch()
