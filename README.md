# 🛡️ Trust Guard AI

**Trust Guard AI** is an advanced, multi-layered content moderation system designed to detect toxicity, hate speech, phishing, and fake news in social media posts. It leverages a hybrid architecture combining traditional sentiment analysis, generative AI, and machine learning.

## 🧠 The 3-Layer Architecture

1.  **Layer 1: Sentiment Analysis (VADER)**
    *   **Role**: Fast, initial scan for negative sentiment.
    *   **Speed**: <10ms.
2.  **Layer 2: Contextual AI (Google Gemini 1.5 Flash)**
    *   **Role**: Deep understanding of sarcasm, hate speech, and **Visual Context** (OCR + Image Analysis).
    *   **Feature**: "Visual Context Detective" reads text inside memes/screenshots to find hidden toxicity.
3.  **Layer 3: Phishing Detector (Scikit-Learn)**
    *   **Role**: Specialized ML classifier (Naive Bayes) trained to detect phishing patterns (e.g., "Urgent", "Verify Account").

## 🚀 Features

*   **📱 Instagram-Style Feed**: Live feed of "Safe" posts.
*   **👮 Admin Dashboard**: Review flagged content with AI reasoning.
*   **🕵️ Visual Context Detective**: Analyzes text *inside* images.
*   **🎣 Phishing Protection**: Auto-blocks suspicious links and urgent scams.
*   **🔄 Feedback Loop**: Admins can approve false positives, which (conceptually) retrains the system.

## 🛠️ Prerequisites

1.  **Python 3.9+**
2.  **Firebase Project**:
    *   Create a project at [console.firebase.google.com](https://console.firebase.google.com/).
    *   Create a **Firestore Database**.
    *   Generate a **Service Account Key** (`serviceAccountKey.json`).
3.  **Google AI Studio Key**:
    *   Get an API key from [aistudio.google.com](https://aistudio.google.com/).

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/trust-guard-ai.git
    cd trust-guard-ai
    ```

2.  **Setup Credentials**
    *   Place your `serviceAccountKey.json` in the root folder.
    *   Create a `.env` file:
        ```env
        GOOGLE_API_KEY=your_api_key_here
        ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train Phishing Model** (First run only)
    ```bash
    python train_phishing.py
    ```

## 🏃‍♂️ Usage

### Option 1: Run Locally (Manual)
You need two terminal windows:

**Terminal 1 (Backend)**:
```bash
uvicorn main:app --reload
```

**Terminal 2 (Frontend)**:
```bash
streamlit run app.py
```

### Option 2: Run with Docker Compose (Recommended)
```bash
docker-compose up --build
```
Access the app at `http://localhost:8501`.

## 📂 Project Structure

*   `app.py`: Streamlit Frontend (UI).
*   `main.py`: FastAPI Backend (API).
*   `guardian.py`: The "Brain" containing the 3-layer logic.
*   `train_phishing.py`: Script to train the ML model.
*   `requirements.txt`: Python dependencies.
