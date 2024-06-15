import streamlit as st
import numpy as np
import cv2
from functions import resize_pad, extract_features
import joblib

try:
  with open('svm_model.pkl', 'rb') as f:
    clf = joblib.load(f)
except FileNotFoundError:
  st.error("Trained model not found. Please train the model first.")
  exit()
  
  
def main():
  """ 
  Main function for the Streamlit app
  """
  # Title and description
  st.title("Handwritten Digit Classifier")
  st.write("Upload an image of a handwritten digit for prediction.")

  # File uploader for user to select an image
  uploaded_file = st.file_uploader("Choose an image...", type=["png"])
  
  if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    # Preprocess the image
    preprocessed_image = resize_pad(image)
    
    # Extract features
    new_features = extract_features(preprocessed_image)
    new_features = np.array([new_features])  # Reshape for prediction

    # Make prediction using the loaded model (if available)
    if clf is not None:
      predicted_label = clf.predict(new_features)[0]
      st.success(f"Predicted digit: {predicted_label}")
    else:
      st.warning("Model not loaded. Please train the model first.")
  
if __name__ == '__main__':
  main()
