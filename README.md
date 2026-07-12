# Voice-Based Parkinson's Disease Detection

This project predicts the likelihood of Parkinson's disease from a person's voice using a Support Vector Machine (SVM). A Streamlit web application is used to upload or record voice samples and display the prediction.

## Technologies

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Librosa
- Parselmouth

## Model

- Algorithm: Support Vector Machine (SVM)
- Training Accuracy: 92.31%
- Testing Accuracy: 82.05%
- Dataset: 195 voice samples

## Files

- website.py - Streamlit application
- engine.py - Model training and prediction
- audio.py - Voice feature extraction
- guardrail.py - Input validation

## Running the Project

```bash
pip install -r requirements.txt
streamlit run website.py
```

## Note

This project was developed for educational purposes and should not be used as a substitute for professional medical diagnosis.
