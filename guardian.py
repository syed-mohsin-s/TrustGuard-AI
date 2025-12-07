import google.generativeai as genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import os
from dotenv import load_dotenv
from PIL import Image
import io
from google.cloud import firestore
import joblib
import numpy as np

# CONFIGURATION
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=api_key)

class GuardianModel:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.gemini = genai.GenerativeModel('gemini-2.5-flash')
        
        # Load Phishing Model (Layer 3)
        try:
            self.phishing_model = joblib.load('phishing_model.pkl')
            print("✅ Phishing Model Loaded")
        except:
            print("⚠️ Phishing Model NOT Found. Layer 3 disabled.")
            self.phishing_model = None

    def analyze(self, text: str, image_blob=None):
        # --- LAYER 1: VADER (Sentiment) ---
        sentiment = self.vader.polarity_scores(text)
        
        # --- LAYER 2: GEMINI (Contextual Understanding) ---
        prompt = f"""
        Analyze this social media content. 
        Text Caption: "{text}"
        
        Task:
        1. **Visual Context Detective**: If an image is provided, extract ALL text from it (OCR).
        2. **Cross-Reference**: Compare the image text with the caption. Look for contradictions or hidden meanings.
        3. **Detect Hate Speech/Cyberbullying**: Be strict. Check both the caption and the image text.
        4. **Detect Sarcasm**: If the text is negative but the image is clearly a joke (meme format), flag as sarcasm.
        5. **Assess Fake News (CRITICAL)**: 
           - Does the image contain statistics, news headlines, or quotes?
           - **FACT CHECK THEM**. If the image claims something factually false (e.g., "The moon is made of cheese"), flag it as HIGH fake_news_likelihood.
           - If the image looks like a manipulated screenshot (fake tweet), flag it.
        
        Return ONLY valid JSON:
        {{
            "is_hate_speech": bool,
            "is_cyberbullying": bool,
            "is_sarcasm": bool,
            "fake_news_likelihood": "LOW" | "MEDIUM" | "HIGH",
            "image_text_content": "Extracted text from image (if any)",
            "reasoning": "Explain your verdict. If Fake News, explicitly state WHY it is false."
        }}
        """
        
        try:
            content = [prompt]
            if image_blob:
                content.append(image_blob)
                
            response = self.gemini.generate_content(content)
            clean_text = response.text.replace('```json', '').replace('```', '').strip()
            ai_analysis = json.loads(clean_text)
            
        except Exception as e:
            ai_analysis = {
                "is_hate_speech": False, 
                "reasoning": f"AI Analysis Failed: {str(e)}",
                "error": True
            }

        # --- LAYER 3: PHISHING DETECTION (Scikit-Learn) ---
        is_phishing = False
        if self.phishing_model:
            # Predict returns [0] or [1]
            pred = self.phishing_model.predict([text])[0]
            if pred == 1:
                is_phishing = True
                ai_analysis['reasoning'] += " [⚠️ PHISHING DETECTED by Layer 3]"

        # --- LOGIC AGGREGATION ---
        final_verdict = "SAFE"
        
        if ai_analysis.get('is_hate_speech') or ai_analysis.get('is_cyberbullying'):
            final_verdict = "FLAGGED"
        elif is_phishing:
             final_verdict = "FLAGGED" # Phishing is dangerous
        elif ai_analysis.get('fake_news_likelihood') == "HIGH":
            final_verdict = "WARNING"
        elif sentiment['compound'] < -0.6 and not ai_analysis.get('is_sarcasm'):
            final_verdict = "REVIEW"
            
        return {
            "text": text,
            "sentiment_score": sentiment['compound'],
            "ai_details": ai_analysis,
            "verdict": final_verdict,
            "timestamp": firestore.SERVER_TIMESTAMP

        }
