# 🐔 Panopticon AI Avian Disease Diagnostics

An AI-powered poultry disease detection system for Pakistan's commercial 
broiler industry. Classifies fecal sample images into 4 disease categories 
using a fine-tuned MobileNetV2 model.

## Live Application
🔗 https://huggingface.co/spaces/MuqsitAlam/AI-Project

## Model Performance
- Architecture: MobileNetV2 
- Test Accuracy: 96.42%
- Classes: Coccidiosis, Healthy, NewCastle, Salmonella

## Repository Structure
- `frontend/` — Streamlit web application
- `backend/` — FastAPI REST backend
- `model/` — Training notebook & saved weights
- `dataset/` — Dataset reference
- `poster/` — Project poster

## Dataset
Sourced from Kaggle — Poultry Disease Dataset  
Link: [(https://www.kaggle.com/datasets/allandclive/chicken-disease-1)
 https://www.kaggle.com/datasets/muhammadmaazsayyed/chicken-disease-dataset
  https://www.kaggle.com/datasets/kausthubkannan/poultry-diseases-detection]

## Tech Stack
Python · TensorFlow · Keras · Streamlit · FastAPI · HuggingFace Spaces
