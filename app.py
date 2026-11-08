from io import BytesIO
import json
import os
import wave
import pandas as pd
import plotly.express as px
import qrcode
import streamlit as st
from groq import Groq

# Page Config
st.set_page_config(page_title="Customer Complaint Analyzer", page_icon="🔍", layout="wide")

# Header & Branding
st.title("🔍 Customer Complaint Analyzer (Cloud AI)")
st.markdown(
    "> **\"निंदक नियरे राखिये, आंगन कुटी छवाय।**\n"
    "> **बिन पानी, साबुन बिना, निर्मल करे सुहाय॥\"**\n\n"
    "*— संत कबीर दास*"
)
st.write(
    "Analyze customer complaints via text or voice (English, Hinglish & Indic languages), log data to CSV,"
    " and monitor real-time trends."
)


# --- PASSWORD AUTHENTICATION ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def analyze_text_with_quota(prompt_text):
    raw_response = client.chat.completions.with_raw_response.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt_text}],
        response_format={"type": "json_object"}
    )
    
    headers = raw_response.headers
    if "x-ratelimit-limit-requests" in headers:
        st.session_state.limit_req = int(headers.get("x-ratelimit-limit-requests", 1000))
    if "x-ratelimit-remaining-requests" in headers:
        st.session_state.remaining_req = int(headers.get("x-ratelimit-remaining-requests", 1000))
    if "x-ratelimit-limit-tokens" in headers:
        st.session_state.limit_tokens = int(headers.get("x-ratelimit-limit-tokens", 6000))
    if "x-ratelimit-remaining-tokens" in headers:
        st.session_state.remaining_tokens = int(headers.get("x-ratelimit-remaining-tokens", 6000))
        
    return json.loads(raw_response.parse().choices[0].message.content)

def verify_password():
    if st.text_input("Password", type="password", key="password_input") == "secret123":
        st.session_state.authenticated = True
        st.rerun()
    else:
        st.error("Incorrect password. Please try again.")

if not st.session_state.authenticated:
    st.markdown("## 🔒 Protected App")
    st.info("Please enter the password to access the Customer Complaint Analyzer.")
    verify_password()
    st.stop()

# --- DYNAMIC GROQ API KEY SETUP ---
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    with st.sidebar:
        st.header("🔑 API Configuration")
        api_key = st.text_input("Enter Groq API Key", type="password")
        if not api_key:
            st.warning("Please enter your Groq API key in the sidebar to enable AI analysis.")
            st.stop()

client = Groq(api_key=api_key)

# Initialize Groq Quota State Defaults
if "limit_req" not in st.session_state:
    st.session_state.limit_req = 1000
if "remaining_req" not in st.session_state:
    st.session_state.remaining_req = 1000
if "remaining_tokens" not in st.session_state:
    st.session_state.remaining_tokens = 6000
if "limit_tokens" not in st.session_state:
    st.session_state.limit_tokens = 6000

# Initialize Session State Variables
LOG_FILE = "complaints_log.csv"

if "history" not in st.session_state:
    if os.path.exists(LOG_FILE):
        try:
            st.session_state.history = pd.read_csv(LOG_FILE).to_dict("records")
        except Exception:
            st.session_state.history = []
    else:
        st.session_state.history = []

if "complaint_text_input" not in st.session_state:
    st.session_state.complaint_text_input = ""

# Quota Tracking Defaults
if "limit_req" not in st.session_state:
    st.session_state.limit_req = 1000
if "remaining_req" not in st.session_state:
    st.session_state.remaining_req = 1000
if "remaining_tokens" not in st.session_state:
    st.session_state.remaining_tokens = 6000
if "limit_tokens" not in st.session_state:
    st.session_state.limit_tokens = 6000

# --- SIDEBAR & CONTROLS ---
with st.sidebar:
    st.header("⚡ Live Groq API Quota")
    used_req = st.session_state.limit_req - st.session_state.remaining_req
    req_ratio = min(max(used_req / max(st.session_state.limit_req, 1), 0.0), 1.0)
    
    st.caption("Daily Requests Remaining")
    st.progress(
        1.0 - req_ratio, 
        text=f"{st.session_state.remaining_req:,} / {st.session_state.limit_req:,} reqs"
    )
    st.caption(f"**Tokens Left (per min):** {st.session_state.remaining_tokens:,}")
    
    st.divider()
    st.header("🔍 Search Archive")
    search_query = st.text_input("Search complaints log", placeholder="Filter by keyword...")

    st.divider()
    st.header("📱 App Access")
    app_url = "https://share.streamlit.io" 
    qr = qrcode.QRCode(version=1, box_size=5, border=2)
    qr.add_data(app_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf)
    st.image(buf.getvalue(), caption="Scan to open app", use_container_width=True)

    st.divider()
    st.header("⚙️ Data Management")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button(
                label="📥 Download CSV Log",
                data=f,
                file_name="complaints_log.csv",
                mime="text/csv",
            )

    if st.button("Clear Analytics & Reset CSV"):
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        st.session_state.history = []
        st.session_state.complaint_text_input = ""
        st.success("CSV log cleared.")
        st.rerun()

# --- AUDIO RECORDING SECTION (MAX 30s) ---
st.subheader("🎙️ Record Complaint by Voice (Max 30s)")
audio_file = st.audio_input("Record your complaint")

transcribed_audio_text = ""
if audio_file is not None:
    audio_bytes = audio_file.getvalue()
    
    # Calculate audio duration (WAV inspect with fallback approximation)
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as wf:
            duration_seconds = wf.getnframes() / float(wf.getframerate())
    except Exception:
        duration_seconds = len(audio_bytes) / 32000.0  # Approx for uncompressed audio stream

    if duration_seconds > 30.0:
        st.error(f"⚠️ Audio exceeds limit! Your recording was {duration_seconds:.1f}s long. Please keep it under 30 seconds.")
    else:
        st.audio(audio_file)
        with st.spinner("Transcribing audio with Groq Whisper..."):
            try:
                transcription = client.audio.transcriptions.create(
                    file=(audio_file.name, audio_bytes),
                    model="whisper-large-v3-turbo",
                    response_format="json"
                )
                transcribed_audio_text = transcription.text
                st.success(f"Audio transcribed successfully ({duration_seconds:.1f}s)!")
            except Exception as e:
                st.error(f"Error transcribing audio: {e}")

# Pre-fill or combine text input
initial_text = transcribed_audio_text if transcribed_audio_text else st.session_state.complaint_text_input

complaint_text = st.text_area(
    "Customer Complaint Text:",
    value=initial_text,
    height=140,
    placeholder="Type your complaint or use the mic recorder above to speak...",
)

# --- DISPLAY SEARCH RESULTS IN MAIN WINDOW ---
if search_query and st.session_state.history:
    st.divider()
    st.subheader(f"🔎 Search Results for '{search_query}'")
    df_history = pd.DataFrame(st.session_state.history)
    mask = df_history.astype(str).apply(lambda x: x.str.contains(search_query, case=False)).any(axis=1)
    filtered_df = df_history[mask]
    st.dataframe(filtered_df, use_container_width=True)