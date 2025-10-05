# Swabhav – AI Personality Predictor

Swabhav is an easy-to-use web application that predicts your MBTI personality type using modern machine learning and psychological insights. The name "Swabhav" means **nature** or **disposition** in Hindi/Gujarati, reflecting our deep focus on understanding your unique personality.

## Features

- **Discover Your Personality:** Instantly predict your MBTI type by answering a short set of questions.
- **State-of-the-Art ML:** Uses a deep neural network trained on real survey data for highly accurate results.
- **Intuitive UI:** Modern design with clear feature explanations and interactive sliders.
- **Detailed Insights:** See your predicted MBTI type along with a confidence breakdown and descriptive interpretation.

## How It Works

1. **Fill Out the Questionnaire:** Enter your age, gender, education, interest, and four key trait scores.
2. **Click "Predict":** The AI model analyzes your data and predicts your MBTI type.
3. **Gain Insights:** See a detailed description of your personality type along with confidence scores for all 16 MBTI types.

## Technologies Used

- **Flask** – Python web framework for backend
- **TensorFlow/Keras** – Deep learning framework for MBTI prediction
- **Joblib** – For model and preprocessor serialization
- **HTML/CSS/Jinja2** – Interactive, responsive frontend
- **Hosted on Render** – Fast and free cloud hosting

## Quick Start
### Run locally (for developers)
```
git clone https://github.com/Vrajesh-Sharma/swabhav.git
cd swabhav
pip install -r requirements.txt
python app.py
```

**Note:** You’ll need the trained model files (`deep_mbti_model.h5`, `scaler.joblib`, encoder files) in the project directory.

## Project Structure
```
swabhav/
├── app.py
├── mbti-personality.ipynb
├── requirements.txt
├── README.md
├── static/
│ ├── style.css
│ └── favicon.png
├── templates/
│ └── index.html
├── model
│ └── deep_mbti_model.h5
├── preprocessing
│ └── scaler.joblib
│ └── le_gender.joblib
│ └── le_edu.joblib
│ └── le_interest.joblib
│ └── le_personality.joblib
```
## Credits

Developed by Vrajesh Sharma  
B.Tech Computer Science, Adani University  
Special thanks to open-source contributors and the MBTI community!

---

**Swabhav: Discover Yourself, Deeply.**
