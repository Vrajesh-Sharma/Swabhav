from flask import Flask, render_template, request
import joblib
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load model & preprocessors
scaler = joblib.load('preprocessing/scaler.joblib')
le_gender = joblib.load('preprocessing/le_gender.joblib')
le_edu = joblib.load('preprocessing/le_edu.joblib')
le_interest = joblib.load('preprocessing/le_interest.joblib')
le_personality = joblib.load('preprocessing/le_personality.joblib')
deep_model = tf.keras.models.load_model('model/deep_mbti_model.h5')

mbti_descriptions = {
    'ENTJ': 'The Commander: Strategic leaders, motivated to organize change',
    'INTJ': 'The Mastermind: Analytical problem-solvers, eager to improve systems and processes',
    'ENTP': 'The Visionary: Inspired innovators, seeking new solutions to challenging problems',
    'INTP': 'The Architect: Philosophical innovators, fascinated by logical analysis',
    'ENFJ': 'The Teacher: Idealist organizers, driven to do what is best for humanity',
    'INFJ': 'The Counselor: Creative nurturers, driven by a strong sense of personal integrity',
    'ENFP': 'The Champion: People-centered creators, motivated by possibilities and potential',
    'INFP': 'The Healer: Imaginative idealists, guided by their own values and beliefs',
    'ESTJ': 'The Supervisor: Hardworking traditionalists, taking charge to get things done',
    'ISTJ': 'The Inspector: Responsible organizers, driven to create order out of chaos',
    'ESFJ': 'The Provider: Conscientious helpers, dedicated to their duties to others',
    'ISFJ': 'The Protector: Industrious caretakers, loyal to traditions and institutions',
    'ESTP': 'The Dynamo: Energetic thrillseekers, ready to push boundaries and dive into action',
    'ISTP': 'The Craftsperson: Observant troubleshooters, solving practical problems',
    'ESFP': 'The Entertainer: Vivacious entertainers, loving life and charming those around them',
    'ISFP': 'The Composer: Gentle caretakers, enjoying the moment with low-key enthusiasm'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    mbti_pred, class_probs, description = None, None, None
    if request.method == 'POST':
        try:
            # Extract and transform form input
            age = float(request.form['age'])
            gender = le_gender.transform([request.form['gender']])[0]
            education = le_edu.transform([request.form['education']])[0]
            interest = le_interest.transform([request.form['interest']])[0]
            introversion = float(request.form['introversion'])
            sensing = float(request.form['sensing'])
            thinking = float(request.form['thinking'])
            judging = float(request.form['judging'])
            features = np.array([[age, gender, education, introversion, sensing, thinking, judging, interest]])
            features_scaled = scaler.transform(features)
            # Predict
            pred_proba = deep_model.predict(features_scaled)
            pred_class = np.argmax(pred_proba)
            mbti_pred = le_personality.inverse_transform([pred_class])[0]
            class_probs = dict(zip(le_personality.classes_, pred_proba[0].round(3)))
            description = mbti_descriptions.get(mbti_pred, '')
        except Exception as e:
            mbti_pred = None
            class_probs = None
            description = f"Error: {e}"

    return render_template(
        'index.html',
        mbti_pred=mbti_pred,
        class_probs=class_probs,
        description=description,
        gender_classes=le_gender.classes_,
        edu_classes=le_edu.classes_,
        interest_classes=le_interest.classes_
    )

if __name__ == '__main__':
    app.run(debug=True)