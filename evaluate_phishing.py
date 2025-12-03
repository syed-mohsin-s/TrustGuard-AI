import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

# 1. Re-create the Dataset (Expanded slightly for better evaluation)
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
        "Your account will be locked. Click to unlock.",
        "Free gift card inside! Open now.",
        "Suspicious activity detected. Confirm it's you.",
        "Final notice: Payment due immediately.",
        
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
        "Please review the attached document.",
        "Are we still on for dinner?",
        "Thanks for your help yesterday.",
        "The package arrived safely.",
        "Call me when you get a chance."
    ],
    'label': [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # 14 Phishing
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # 14 Safe
    ]
}

df = pd.DataFrame(data)

# 2. Build the Pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 3. Evaluate using 5-Fold Cross-Validation
# This splits the data into 5 parts, training on 4 and testing on 1, five times.
scores = cross_val_score(model, df['text'], df['label'], cv=5, scoring='accuracy')

print(f"Individual Fold Scores: {scores}")
print(f"Average Accuracy: {np.mean(scores) * 100:.2f}%")
