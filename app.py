from io import BytesIO
import json
import os
import re
import pandas as pd
import plotly.express as px
import qrcode
import speech_recognition as sr
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from groq import Groq

# Page Config
st.set_page_config(
    page_title="Customer Complaint Analyzer",
    page_icon="🔍",
    layout="wide",
)

# --- PASSWORD AUTHENTICATION ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

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
api_key = None
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    with st.sidebar:
        st.header("🔑 API Configuration")
        api_key = st.text_input("Enter Groq API Key", type="password")
        if not api_key:
            st.warning("Please enter your Groq API key in the sidebar to enable AI analysis.")
            st.stop()

client = Groq(api_key=api_key)

LOG_FILE = "complaints_log.csv"

st.title("🔍 Customer Complaint Analyzer (Cloud AI)")
st.write(
    "Analyze customer complaints via text or voice, log data to CSV,"
    " and monitor real-time trends."
)

# Initialize Session State
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

# Sidebar: QR Code & History Controls
with st.sidebar:
  st.header("📱 App Access")
  app_url = "https://share.streamlit.io" 
  qr = qrcode.QRCode(version=1, box_size=5, border=2)
  qr.add_data(app_url)
  qr.make(fit=True)
  img = qr.make_image(fill_color="black", back_color="white")
  buf = BytesIO()
  img.save(buf)
  st.image(
      buf.getvalue(), caption="Scan to open app", use_container_width=True
  )

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

# Voice Recording Section
st.subheader("🎙️ Record Complaint by Voice")
audio = mic_recorder(
    start_prompt="🔴 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    key="recorder",
)

if audio:
  with st.spinner("Transcribing audio..."):
    recognizer = sr.Recognizer()
    audio_data = sr.AudioData(
        audio["bytes"], sample_rate=16000, sample_width=2
    )
    try:
      transcribed_text = recognizer.recognize_google(audio_data)
      st.session_state.complaint_text_input = transcribed_text
      st.success("Voice transcribed successfully!")
    except sr.UnknownValueError:
      st.error("Could not understand the audio. Please try speaking clearer.")
    except sr.RequestError as e:
      st.error(f"Speech recognition service error: {e}")

# Text Area connected to session state
complaint_text = st.text_area(
    "Customer Complaint Text:",
    value=st.session_state.complaint_text_input,
    height=140,
    placeholder=(
        "Type your complaint or use the mic button above to speak..."
    ),
)

if st.button("Analyze Complaint", type="primary"):
  if not complaint_text.strip():
    st.warning("Please paste or record a complaint to analyze.")
  else:
    with st.spinner("Analyzing complaint with Groq (Llama 3)..."):
      prompt = f"""
            Analyze the following customer complaint and extract structured information.
            Return ONLY a valid JSON object matching this exact schema:
            {{
                "category": "String (e.g., Product Quality, Shipping Delay, Billing, Support Experience)",
                "urgency": "String (Low, Medium, High, Critical)",
                "sentiment": "String (Negative, Extremely Frustrated, Neutral)",
                "key_issues": ["List of main problems identified"],
                "summary": "Concise 1-2 sentence summary of the issue",
                "recommended_action": "Suggested response or step for the support agent to take"
            }}

            Complaint:
            "{complaint_text}"
            """
      try:
        chat_completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = chat_completion.choices[0].message.content

        # Robust JSON Extraction
        try:
            cleaned_content = content
            if "```json" in cleaned_content:
                cleaned_content = cleaned_content.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_content:
                cleaned_content = cleaned_content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(cleaned_content)
        except Exception:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                raise ValueError(f"Model did not return valid JSON. Raw output: {content}")

        record = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "complaint": complaint_text,
            "category": result.get("category", "Unknown"),
            "urgency": result.get("urgency", "Low"),
            "sentiment": result.get("sentiment", "Neutral"),
            "summary": result.get("summary", ""),
            "key_issues": ", ".join(result.get("key_issues", [])),
            "recommended_action": result.get("recommended_action", ""),
        }

        # Save record to local CSV
        df_new = pd.DataFrame([record])
        if not os.path.exists(LOG_FILE):
          df_new.to_csv(LOG_FILE, index=False)
        else:
          df_new.to_csv(LOG_FILE, mode="a", header=False, index=False)

        st.session_state.history.append(record)

        # Display Current Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Category", result.get("category", "N/A"))
        col2.metric("Urgency Level", result.get("urgency", "N/A"))
        col3.metric("Customer Sentiment", result.get("sentiment", "N/A"))

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
          st.subheader("📌 Summary")
          st.write(result.get("summary", ""))

          st.subheader("❗ Key Issues Identified")
          for issue in result.get("key_issues", []):
            st.write(f"- {issue}")

        with col_b:
          st.subheader("💡 Recommended Action")
          st.info(result.get("recommended_action", ""))

      except Exception as e:
        st.error(f"An error occurred: {e}")

# Render Collapsible History & Dashboard
if st.session_state.history:
  st.divider()
  st.header("📊 Analytics & History Dashboard")

  df = pd.DataFrame(st.session_state.history)

  c1, c2 = st.columns(2)
  with c1:
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]
    fig_sentiment = px.pie(
        sentiment_counts,
        names="sentiment",
        values="count",
        title="Overall Sentiment Distribution",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

  with c2:
    urgency_order = ["Low", "Medium", "High", "Critical"]
    urgency_counts = (
        df["urgency"]
        .value_counts()
        .reindex(urgency_order, fill_value=0)
        .reset_index()
    )
    urgency_counts.columns = ["urgency", "count"]
    fig_urgency = px.bar(
        urgency_counts,
        x="urgency",
        y="count",
        title="Urgency Distribution",
        color="urgency",
        color_discrete_map={
            "Low": "#2ecc71",
            "Medium": "#f1c40f",
            "High": "#e67e22",
            "Critical": "#e74c3c",
        },
    )
    st.plotly_chart(fig_urgency, use_container_width=True)

  # Collapsible History Views
  with st.expander("📊 View Complete Logged Records (Table View)", expanded=False):
    st.dataframe(df, use_container_width=True)

  with st.expander("📂 View Individual Past Complaints", expanded=False):
    for record in reversed(st.session_state.history):
      header_title = (
          f"[{record['timestamp']}] {record['category']} | Urgency:"
          f" {record['urgency']} | Sentiment: {record['sentiment']}"
      )
      with st.expander(header_title, expanded=False):
        col_l, col_r = st.columns(2)
        with col_l:
          st.markdown("**Original Complaint:**")
          st.write(record["complaint"])
          st.markdown("**Summary:**")
          st.write(record["summary"])
        with col_r:
          st.markdown("**Key Issues:**")
          st.write(record["key_issues"])
          st.markdown("**Recommended Action:**")
          st.info(record["recommended_action"])