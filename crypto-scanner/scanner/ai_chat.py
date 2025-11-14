# scanner/ai_chat.py
import os
import json
import sqlite3
import requests
import streamlit as st
from datetime import datetime
from scanner.sentiment_binance import fetch_binance_sentiment



# ---------------------------------------------------------------------
# ---------------------- KONFIGURASI DASAR ----------------------------
# ---------------------------------------------------------------------

OLLAMA_API = "http://localhost:11434/api/generate"
DB_FILE = "data/ai_memory.db"


# ---------------------------------------------------------------------
# ---------------------- DATABASE (MEMORY) -----------------------------
# ---------------------------------------------------------------------

def init_db():
    """Inisialisasi database SQLite untuk menyimpan chat history."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def load_memory(user_id="default"):
    """Ambil history percakapan user dari database."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content, timestamp FROM memory WHERE user_id=? ORDER BY id ASC LIMIT 50",
        (user_id,)
    )
    data = cur.fetchall()
    conn.close()

    history = []
    current_pair = {}
    for role, text, ts in data:
        if role == "user":
            current_pair = {"user": text, "ai": "", "timestamp": ts}
            history.append(current_pair)
        elif role == "ai":
            if history:
                history[-1]["ai"] = text
                history[-1]["timestamp"] = ts
    return history


def save_message(role, text, user_id="default"):
    """Simpan satu pesan (user/AI) ke database."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO memory (user_id, role, content) VALUES (?, ?, ?)",
        (user_id, role, text)
    )
    conn.commit()
    conn.close()


def clear_memory(user_id="default"):
    """Hapus semua chat history user."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM memory WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# ---------------------- MODEL KOMUNIKASI -----------------------------
# ---------------------------------------------------------------------

def chat_with_deepseek(prompt: str, history: list, model="deepseek-coder:1.3b"):
    """Kirim prompt ke DeepSeek lokal via Ollama."""
    sentiment = fetch_binance_sentiment("BTCUSDT")

    market_context = f"""
KONDISI PASAR TERKINI:
- BTC Funding Rate: {sentiment.get('funding_rate', 'n/a')}%
- BTC Open Interest: {sentiment.get('open_interest', 0)/1e6:.2f}M USDT
- BTC Long/Short Ratio: {sentiment.get('long_short_ratio', 'n/a')}
"""

    past = "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in history[-8:]])

    full_prompt = f"""{market_context}

Kamu adalah analis profesional pasar kripto.
Gunakan analisis teknikal dan reasoning logis untuk menjawab pertanyaan berikut.

{past}
User: {prompt}
AI:"""

    payload = {"model": model, "prompt": full_prompt, "stream": True}

    try:
        with requests.post("http://localhost:11434/api/generate", json=payload, stream=True, timeout=180) as r:
            r.raise_for_status()
            full_response = ""
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        full_response += data["response"]
                except json.JSONDecodeError:
                    continue
            return full_response.strip()
    except Exception as e:
        return f"[Error] Tidak bisa terhubung ke model lokal DeepSeek: {e}"




# ---------------------------------------------------------------------
# ---------------------- STREAMLIT UI ---------------------------------
# ---------------------------------------------------------------------

def render_chat_tab():
    """Render tab Chat AI dengan UI seperti ChatGPT, scrollable & persistent memory."""
    st.header("🤖 Qwen AI Analyst (Local + Persistent Memory)")
    st.caption("Model Qwen2-1.5B berjalan di CPU lokal, dengan memori percakapan dan konteks pasar real-time.")

    init_db()
    user_id = "default"

    # Ambil history dari DB
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_memory(user_id)

    # ======================== AREA CHAT SCROLLABLE =========================
    chat_html = """
    <div id='chat-box' style='max-height:500px; overflow-y:auto; padding:16px;
         border-radius:12px; background-color:#111; border:1px solid #333;
         font-family: monospace; scroll-behavior:smooth;'>
    """

    for h in st.session_state.chat_history:
        timestamp = h.get("timestamp", datetime.now().strftime("%H:%M"))
        user_msg = h["user"].replace("\n", "<br>")
        ai_msg = h["ai"].replace("\n", "<br>")

        chat_html += f"""
        <div style='margin-bottom:14px;'>
            <div style='text-align:right;'>
                <div style='display:inline-block; background-color:#2c2f3a; color:#9cf;
                    padding:8px 12px; border-radius:12px; max-width:75%; word-wrap:break-word;'>
                    {user_msg}
                </div>
                <div style='color:#777; font-size:11px; margin-top:2px;'>{timestamp}</div>
            </div>

            <div style='text-align:left; margin-top:6px;'>
                <div style='display:inline-block; background-color:#1a1a1a; color:#9f9;
                    padding:8px 12px; border-radius:12px; max-width:80%; word-wrap:break-word;'>
                    {ai_msg}
                </div>
                <div style='color:#555; font-size:11px; margin-top:2px;'>{timestamp}</div>
            </div>
        </div>
        """

    chat_html += "</div>"

    # Render HTML bubble chat
    #st.markdown(chat_html, unsafe_allow_html=True)
    import streamlit.components.v1 as components
    components.html(
        f"""
        <html>
        <head>
            <meta charset='utf-8'>
            <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: #111;
                color: #ddd;
                font-family: monospace;
            }}
            #chat-box {{
                max-height: 500px;
                overflow-y: auto;
                padding: 16px;
                border-radius: 12px;
                background-color: #111;
                border: 1px solid #333;
                scroll-behavior: smooth;
            }}
            </style>
        </head>
        <body>
            {chat_html}
            <script>
            // Auto scroll ke bawah tiap render ulang
            const chatBox = document.getElementById('chat-box');
            if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
            </script>
        </body>
        </html>
        """,
        height=550,
        scrolling=True,
    )
    
    # Auto-scroll ke bawah
    st.markdown("""
        <script>
        const chatBox = window.parent.document.querySelector('#chat-box');
        if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
        </script>
    """, unsafe_allow_html=True)

    # ======================== INPUT AREA =========================
    st.markdown("---")
    query = st.text_area("💬 Ketik pertanyaan atau diskusi market:", key="user_input")

    col1, col2 = st.columns([1, 0.3])
    with col1:
        send_clicked = st.button("🚀 Kirim", use_container_width=True)
    with col2:
        clear_clicked = st.button("🧹 Hapus Semua", use_container_width=True)

    if send_clicked:
        if query.strip():
            with st.spinner("Qwen sedang menganalisis..."):
               #response = chat_with_qwen(query, st.session_state.chat_history)
                response = chat_with_deepseek(query, st.session_state.chat_history)
            timestamp = datetime.now().strftime("%H:%M")
            # Simpan ke DB dan session
            save_message("user", query, user_id)
            save_message("ai", response, user_id)
            st.session_state.chat_history.append({
                "user": query,
                "ai": response,
                "timestamp": timestamp
            })
            st.rerun()
        else:
            st.warning("Masukkan pertanyaan dulu.")

    if clear_clicked:
        clear_memory(user_id)
        st.session_state.chat_history = []
        st.rerun()
