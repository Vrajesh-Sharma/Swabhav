from flask import Flask, render_template, request, session, redirect, url_for
import joblib
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "verysecretkey"

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

# Mapping for MCQ answers (A/B/C/D) to numeric scores (customize as needed)
MCQ_MAP = {'A': 10, 'B': 7, 'C': 4, 'D': 1}

# MCQ config: 5 traits with 4 Qs each
TRAITS = [
    ('extroversion', "Extroversion", [
        "Are you comfortable with public speaking?",
        "Do you enjoy being the center of attention?",
        "Do you thrive in group settings?",
        "Do you make friends easily at new places?",
    ]),
    ('sensing', "Sensing", [
        "Do you focus more on immediate details than the big picture?",
        "Do you trust experience over intuition?",
        "Are you comfortable following set routines?",
        "Do you prefer facts to theories?",
    ]),
    ('thinking', "Thinking", [
        "Are you logical when making decisions?",
        "Do you value truth over social harmony?",
        "Do you enjoy debates?",
        "Are you comfortable with criticizing others when needed?",
    ]),
    ('judging', "Judging", [
        "Do you prefer planned activities over spontaneous ones?",
        "Do you enjoy making schedules and to-do lists?",
        "Are you uncomfortable with last-minute changes?",
        "Do you finish assignments ahead of time?",
    ]),
    ('perceiving', "Perceiving", [
        "Are you flexible with plans?",
        "Can you adapt easily to changes?",
        "Do you often start things without finishing?",
        "Do you leave decisions to the last minute?",
    ])
]

# Render demographic info page (step 0)
@app.route('/', methods=['GET', 'POST'])
def demographics():
    if request.method == 'POST':
        session['demographics'] = {
            'age': request.form['age'],
            'gender': request.form['gender'],
            'education': request.form['education'],
            'interest': request.form['interest']
        }
        return redirect(url_for('mcq', step=1))
    return render_template('index.html',
                           gender_classes=le_gender.classes_,
                           edu_classes=le_edu.classes_,
                           interest_classes=le_interest.classes_)

# Render trait-specific MCQ stepper
@app.route('/mcq/<int:step>', methods=['GET', 'POST'])
def mcq(step):
    total_steps = len(TRAITS)
    if request.method == 'POST' and step > 1:
        prev_trait_key = TRAITS[step - 2][0]
        session[prev_trait_key] = [
            request.form.get(f'{prev_trait_key}_q{i}', None) for i in range(1, 5)
        ]
    if step > total_steps:
        return redirect(url_for('results'))
    trait_key, trait_label, questions = TRAITS[step - 1]
    return render_template('mcq_step.html',
                          trait_key=trait_key,
                          trait_label=trait_label,
                          questions=questions,
                          step=step,
                          total_steps=total_steps)

# Compute scores and predict
@app.route('/results')
def results():
    try:
        data = session.get('demographics', {})
        age = float(data['age']) if 'age' in data else 25
        gender = le_gender.transform([data['gender']])[0] if 'gender' in data else 0
        education = le_edu.transform([data['education']])[0] if 'education' in data else 0
        interest = le_interest.transform([data['interest']])[0] if 'interest' in data else 0

        # Only collect trait scores for the 4 expected traits in this order
        trait_keys = ['extroversion', 'sensing', 'thinking', 'judging']
        trait_scores = []
        for trait_key in trait_keys:
            answers = session.get(trait_key, ['C', 'C', 'C', 'C'])  # default 'C'
            numeric_scores = [MCQ_MAP.get(ans, 4) for ans in answers]
            trait_scores.append(float(sum(numeric_scores)) / 4)

        # Assemble the features array as per model expectation (8 features only)
        features = np.array([[age, gender, education] + trait_scores + [interest]])
        features_scaled = scaler.transform(features)
        pred_proba = deep_model.predict(features_scaled)
        pred_class = np.argmax(pred_proba)
        mbti_pred = le_personality.inverse_transform([pred_class])[0]
        class_probs = dict(zip(le_personality.classes_, pred_proba[0].round(3)))
        description = mbti_descriptions.get(mbti_pred, '')
    except Exception as e:
        mbti_pred, class_probs, description = None, None, f"Error: {e}"
    # Clear session for new test
    session.clear()
    return render_template('results.html',
                           mbti_pred=mbti_pred,
                           class_probs=class_probs,
                           description=description)

if __name__ == '__main__':
    app.run(debug=True)