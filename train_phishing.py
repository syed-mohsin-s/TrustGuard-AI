import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# 1. Create a Mock Dataset (In production, load from CSV)
data = {
    'text': [
        # Phishing / Spam
        "Urgent! Verify your account now.",
        "Click this link to claim your prize.",
        "Your bank account has been compromised. Login here.",
        "Win a free iPhone! Click here.",
        "Security Alert: Unusual sign-in activity.",
        "Update your password immediately.",
        "IRS Tax Refund Pending. Claim now.",
        "You have won the lottery! Contact us.",
        "Verify your identity to avoid suspension.",
        "Exclusive offer! Expires in 1 hour.",
        
        # Safe / Normal
        "Hey, how are you doing?",
        "Meeting at 3 PM confirmed.",
        "Can you send me the report?",
        "I love this new song!",
        "Happy Birthday! Have a great day.",
        "Just checking in on the project.",
        "Let's grab lunch tomorrow.",
        "The weather is beautiful today.",
        "Did you see the game last night?",
        "Please review the attached document."
    ],
    'label': [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # 1 = Phishing
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # 0 = Safe
    ]
}

df = pd.DataFrame(data)

# 2. Build the Pipeline
# TF-IDF converts text to numbers, Naive Bayes classifies them
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 3. Train the Model
print("Training Phishing Detector...")
model.fit(df['text'], df['label'])

# 4. Test
test_phish = "Urgent: Update your bank details"
prediction = model.predict([test_phish])[0]
print(f"Test '{test_phish}': {'PHISHING' if prediction == 1 else 'SAFE'}")

# 5. Save the Model
joblib.dump(model, 'phishing_model.pkl')
print("Model saved to phishing_model.pkl")
