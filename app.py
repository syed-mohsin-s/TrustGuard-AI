import streamlit as st
import requests
from google.cloud import firestore
from firebase_admin import credentials, firestore as admin_firestore, initialize_app, get_app
import os

# --- FIREBASE SETUP FOR STREAMLIT ---
# (Singleton pattern to prevent re-initialization errors on reload)
if not os.path.exists("serviceAccountKey.json"):
    st.error("Missing serviceAccountKey.json. Please place it in the root directory.")
    st.stop()

try:
    app = get_app()
except ValueError:
    cred = credentials.Certificate("serviceAccountKey.json")
    initialize_app(cred)
db = admin_firestore.client()

st.set_page_config(layout="wide", page_title="Guardian AI")

# UI Layout
st.title("🛡️ Guardian AI Network")
tab_feed, tab_admin, tab_post = st.tabs(["📱 Live Feed", "👮 Admin Dashboard", "➕ Create Post"])

# --- TAB 1: INSTAGRAM STYLE FEED ---
with tab_feed:
    st.subheader("Global Feed")
    # Fetch posts that are NOT flagged
    posts = db.collection('posts').where('verdict', '==', 'SAFE').order_by('timestamp', direction='DESCENDING').stream()
    
    for post in posts:
        data = post.to_dict()
        with st.container(border=True):
            c1, c2 = st.columns([1, 5])
            with c1:
                st.image("https://api.dicebear.com/7.x/avataaars/svg?seed=" + data['username'], width=50)
            with c2:
                st.write(f"**@{data['username']}**")
                st.write(data['text'])
                st.caption(f"Sentiment: {data['sentiment_score']} | 🤖 Checked by Guardian")

# --- TAB 2: ADMIN DASHBOARD (Feedback Loop) ---
with tab_admin:
    st.subheader("Content Moderation Queue")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    # Real-time stats (simplified for demo)
    flagged_docs = list(db.collection('posts').where('verdict', 'in', ['FLAGGED', 'WARNING']).stream())
    col1.metric("Items in Review", len(flagged_docs))
    col2.metric("System Health", "Online")
    
    st.divider()
    
    for doc in flagged_docs:
        data = doc.to_dict()
        with st.expander(f"🔴 FLAGGED: {data['text'][:30]}..."):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**User:** @{data['username']}")
                st.write(f"**Content:** {data['text']}")
                st.warning(f"**AI Reason:** {data['ai_details'].get('reasoning')}")
                
                # Visualizing the Logic
                st.json(data['ai_details'])
                
            with c2:
                st.write("### Moderator Action")
                # FEEDBACK LOOP: If Admin approves, we override the AI
                if st.button("✅ False Positive - Approve", key=f"app_{doc.id}"):
                    db.collection('posts').document(doc.id).update({'verdict': 'SAFE'})
                    st.rerun()
                    
                if st.button("🚫 Confirm Ban", key=f"ban_{doc.id}"):
                    db.collection('posts').document(doc.id).delete()
                    st.rerun()

# --- TAB 3: POST SIMULATOR ---
with tab_post:
    st.header("Simulate a User Post")
    with st.form("new_post"):
        user = st.text_input("Username", "anonymous_user")
        txt = st.text_area("Caption")
        img_file = st.file_uploader("Upload Image (Optional)", type=['png', 'jpg', 'jpeg'])
        submitted = st.form_submit_button("Post to Feed")
        
        if submitted:
            # Call our own API (Localhost)
            try:
                files = None
                if img_file:
                    files = {"image": (img_file.name, img_file, img_file.type)}
                
                res = requests.post("http://localhost:8000/analyze_post", data={"username": user, "caption": txt}, files=files)
                if res.status_code == 200:
                    verdict = res.json()['verdict']
                    if verdict == "SAFE":
                        st.success("Posted successfully!")
                    else:
                        st.error(f"Post blocked by Guardian! Reason: {verdict}")
                else:
                    st.error("Server Error")
            except:
                st.error("Ensure FastAPI backend is running on port 8000")