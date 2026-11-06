# 💬 WhatsApp Chat Analyzer

A production-grade, modular web application for end-to-end WhatsApp chat parsing, statistical analytics, and NLP-driven behavioral intelligence. Built with **Streamlit**, **Pandas**, **NLTK (VADER)**, and interactive **Plotly** visualizations.

---

## 🚀 Key Features

- **Robust Timestamp Parsing:**
  - Auto-detects and parses Android and iOS formats across both 12-hour (`AM/PM`, unicode spaces `\u202f`) and 24-hour systems.
  - Multi-line message continuity and system event classification (encryption notices, group creation, member joins/exits).
- **Media & Deleted Message Handling:**
  - Identifies media omission placeholders (`<Media omitted>`, `image omitted`, `video omitted`, `document omitted`, stickers).
  - Automatically filters placeholders and deleted messages from textual NLP analysis while tallying them into accurate KPI counts.
- **Statistical Analytics & Activity Mapping:**
  - Top KPIs: Message count, word count, media shared, and URLs extracted.
  - Interactive Monthly & Daily activity timelines.
  - Busiest days of the week, busiest months, and a weekly 24-hour hourly activity heatmap.
- **Group Member Engagement:**
  - Most active contributors ranking with percentage contribution and interactive horizontal bar charts.
- **Linguistic & NLP Exploration:**
  - High-resolution Word Cloud excluding standard English + Hinglish chat colloquialisms.
  - Top 20 most frequent words frequency chart.
  - Emoji parsing and top-5 distribution donut chart using the `emoji` library.
- **Sentiment & Emotional Tone Analysis (VADER):**
  - Polarity scoring (Positive, Neutral, Negative) using NLTK `SentimentIntensityAnalyzer`.
  - User-wise sentiment breakdown and mood summary.
  - Identification of highest positive and critical/negative messages.
- **Cleaned Data Explorer & Export:**
  - In-app searchable data table with instant `.csv` download.

---

## 📂 Project Architecture

```
whatsapp-chat-analyzer/
├── app.py                 # Main Streamlit dashboard application
├── preprocessor.py        # Regex parsing and DataFrame creation engine
├── helper.py              # Statistical metrics, NLP pipelines, and chart builders
├── requirements.txt       # Dependencies (streamlit, pandas, nltk, plotly, wordcloud, emoji)
├── sample_chat.txt        # Built-in sample chat for instant demonstration & testing
└── README.md              # Setup instructions & usage guide
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Web Dashboard** | Streamlit |
| **Data Manipulation** | Pandas, NumPy |
| **NLP & Sentiment** | NLTK (VADER SentimentIntensityAnalyzer), Stopwords, WordCloud |
| **Emoji Processing** | `emoji` library |
| **Interactive Charts** | Plotly Express & Plotly Graph Objects |
| **Static Visualizations** | Matplotlib & Seaborn |
| **Pattern Matching** | Python `re` (Regex) |

---

## 📥 How to Export Chat from WhatsApp

### 📱 Android
1. Open the individual or group chat in WhatsApp.
2. Tap the **three vertical dots (⋮)** in the top right corner.
3. Tap **More** > **Export chat**.
4. Choose **Without Media** *(crucial for standard text export)*.
5. Save or email the resulting `.txt` file to your computer.

### 🍎 iOS (iPhone)
1. Open the individual or group chat in WhatsApp.
2. Tap the contact or group name at the top of the conversation.
3. Scroll down and tap **Export Chat**.
4. Select **Without Media**.
5. Save the file to Files or transfer it via AirDrop / Email.

---

## ⚙️ Installation & Local Execution

### 1. Clone or Navigate to the Directory
```bash
cd whatsapp-chat-analyzer
```

### 2. Create and Activate a Virtual Environment (Recommended)
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 🧪 Testing with Built-in Demo Data

If you do not have an exported WhatsApp chat file on hand, simply launch the app and click the **"📂 Or load sample project chat"** checkbox in the sidebar. This loads a realistic multi-user chat with dates, emojis, links, media tags, and positive/constructive discussions.

---

## 🧩 Regex Pattern Reference

The preprocessing engine leverages the following regex patterns to isolate WhatsApp date headers across Android and iOS exports:

```python
# Android 12h: DD/MM/YYYY, HH:MM AM - Author: message
r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)\s-\s)'

# Android 24h: DD/MM/YYYY, HH:MM - Author: message
r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s)'

# iOS 12h: [DD/MM/YYYY, HH:MM:SS AM] Author: message
r'(\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s?(?:AM|PM|am|pm)\]\s)'

# iOS 24h: [DD/MM/YYYY, HH:MM:SS] Author: message
r'(\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\]\s)'
```
