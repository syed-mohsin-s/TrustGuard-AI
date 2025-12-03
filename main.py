from fastapi import FastAPI, UploadFile, File, Form
from guardian import GuardianModel
import firebase_admin
from firebase_admin import credentials, firestore
from PIL import Image
import io

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = FastAPI()
guardian = GuardianModel()

@app.post("/analyze_post")
async def analyze_post(
    username: str = Form(...), 
    caption: str = Form(...),
    image: UploadFile = File(None)
):
    # 1. Run the Guardian Model
    image_blob = None
    if image:
        image_bytes = await image.read()
        image_blob = Image.open(io.BytesIO(image_bytes))

    result = guardian.analyze(caption, image_blob=image_blob)
    
    # 2. Add Metadata
    result['username'] = username
    
    # 3. Save to Firestore (The Feedback Loop Source)
    # We save EVERYTHING. Even flagged content (so admins can review it).
    doc_ref = db.collection('posts').add(result)
    
    return {"status": "success", "verdict": result['verdict'], "id": doc_ref[1].id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)