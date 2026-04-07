from flask import Flask, request, jsonify, render_template_string
from google import genai
import os

app = Flask(__name__)

# Read the Gemini API key from your environment.
# In PowerShell, set it like this before running the app:
# $env:GEMINI_API_KEY="your_real_key_here"
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. In PowerShell, run: $env:GEMINI_API_KEY=\"your_real_key_here\""
    )

client = genai.Client(api_key=API_KEY)
chat = client.chats.create(model="gemini-2.5-flash")

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>My Gemini Chat</title>
  <style>
    :root {
      --bg-1: #020617;
      --bg-2: #0f172a;
      --panel: rgba(15, 23, 42, 0.82);
      --panel-strong: rgba(15, 23, 42, 0.96);
      --panel-soft: rgba(30, 41, 59, 0.72);
      --text: #e5eefc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-2: #818cf8;
      --user: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
      --bot: rgba(30, 41, 59, 0.92);
      --border: rgba(148, 163, 184, 0.16);
      --shadow: 0 24px 80px rgba(2, 8, 23, 0.45);
    }

    * { box-sizing: border-box; }

    html, body {
      margin: 0;
      min-height: 100%;
      font-family: Inter, Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(129, 140, 248, 0.14), transparent 22%),
        linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
    }

    body {
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 24px;
    }

    .app {
      width: 100%;
      max-width: 980px;
      height: min(88vh, 920px);
      display: flex;
      flex-direction: column;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 28px;
      overflow: hidden;
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }

    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 22px 24px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(15, 23, 42, 0.88));
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }

    .logo {
      width: 46px;
      height: 46px;
      border-radius: 16px;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
      display: grid;
      place-items: center;
      color: white;
      font-size: 22px;
      box-shadow: 0 10px 28px rgba(56, 189, 248, 0.32);
      flex-shrink: 0;
    }

    .title-wrap h1 {
      margin: 0;
      font-size: 27px;
      line-height: 1.1;
      letter-spacing: -0.03em;
    }

    .title-wrap p {
      margin: 5px 0 0;
      color: var(--muted);
      font-size: 14px;
    }

    .badge {
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.1);
      border: 1px solid rgba(56, 189, 248, 0.18);
      color: #bae6fd;
      font-size: 13px;
      white-space: nowrap;
    }

    .chat-box {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      scroll-behavior: smooth;
      background:
        linear-gradient(180deg, rgba(2, 6, 23, 0.18), rgba(2, 6, 23, 0.32));
    }

    .row {
      display: flex;
      align-items: flex-end;
      gap: 12px;
      animation: fadeInUp 0.22s ease;
    }

    .row.user-row {
      justify-content: flex-end;
    }

    .avatar {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      flex-shrink: 0;
      font-size: 16px;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.35);
    }

    .bot-avatar {
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.22), rgba(129, 140, 248, 0.22));
      border: 1px solid rgba(129, 140, 248, 0.18);
    }

    .user-avatar {
      background: linear-gradient(135deg, rgba(14, 165, 233, 0.28), rgba(37, 99, 235, 0.28));
      border: 1px solid rgba(56, 189, 248, 0.18);
      order: 2;
    }

    .bubble-wrap {
      max-width: min(78%, 720px);
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .meta {
      font-size: 12px;
      color: var(--muted);
      padding: 0 8px;
    }

    .message {
      padding: 15px 17px;
      border-radius: 20px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid transparent;
    }

    .bot .message {
      background: var(--bot);
      border-color: var(--border);
      border-bottom-left-radius: 6px;
    }

    .user .message {
      background: var(--user);
      color: white;
      border-bottom-right-radius: 6px;
    }

    .composer {
      padding: 16px 18px 18px;
      border-top: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(15, 23, 42, 0.84), rgba(15, 23, 42, 0.98));
    }

    .input-shell {
      display: flex;
      gap: 12px;
      padding: 10px;
      border-radius: 22px;
      background: rgba(2, 6, 23, 0.42);
      border: 1px solid rgba(56, 189, 248, 0.16);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }

    #messageInput {
      flex: 1;
      resize: none;
      min-height: 58px;
      max-height: 180px;
      padding: 14px 16px;
      border: none;
      border-radius: 16px;
      background: transparent;
      color: var(--text);
      font-size: 15px;
      outline: none;
      font-family: inherit;
    }

    #messageInput::placeholder {
      color: #7c8aa5;
    }

    .send-btn {
      min-width: 112px;
      border: none;
      border-radius: 18px;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
      color: white;
      font-weight: 700;
      font-size: 15px;
      cursor: pointer;
      box-shadow: 0 12px 28px rgba(56, 189, 248, 0.28);
      transition: transform 0.15s ease, filter 0.15s ease, opacity 0.15s ease;
    }

    .send-btn:hover {
      transform: translateY(-1px);
      filter: brightness(1.04);
    }

    .send-btn:disabled {
      opacity: 0.7;
      cursor: not-allowed;
      transform: none;
    }

    .footer-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-top: 10px;
      padding: 0 4px;
      color: var(--muted);
      font-size: 12px;
    }

    .typing-row {
      display: none;
      align-items: center;
      gap: 12px;
      animation: fadeInUp 0.22s ease;
    }

    .typing-pill {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 12px 14px;
      border-radius: 18px;
      background: var(--bot);
      border: 1px solid var(--border);
      border-bottom-left-radius: 6px;
    }

    .dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #cbd5e1;
      opacity: 0.65;
      animation: bounce 1.1s infinite ease-in-out;
    }

    .dot:nth-child(2) { animation-delay: 0.15s; }
    .dot:nth-child(3) { animation-delay: 0.3s; }

    @keyframes bounce {
      0%, 80%, 100% { transform: translateY(0); opacity: 0.45; }
      40% { transform: translateY(-4px); opacity: 1; }
    }

    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 720px) {
      body { padding: 10px; }
      .app { height: 94vh; border-radius: 22px; }
      .header { padding: 18px; align-items: flex-start; flex-direction: column; }
      .badge { align-self: flex-start; }
      .chat-box { padding: 18px; }
      .bubble-wrap { max-width: 88%; }
      .input-shell { flex-direction: column; }
      .send-btn { min-height: 52px; }
      .footer-row { flex-direction: column; align-items: flex-start; }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="header">
      <div class="brand">
        <div class="logo">✦</div>
        <div class="title-wrap">
          <h1>My Gemini Chat</h1>
          <p>A tiny local AI website built with Python + Flask + Gemini.</p>
        </div>
      </div>
      <div class="badge">Remembers while open</div>
    </div>

    <div id="chatBox" class="chat-box">
      <div class="row bot-row bot">
        <div class="avatar bot-avatar">🤖</div>
        <div class="bubble-wrap">
          <div class="meta">Gemini</div>
          <div class="message">Hi! I remember earlier messages while this app is running. Ask me anything.</div>
        </div>
      </div>

      <div id="typingRow" class="typing-row">
        <div class="avatar bot-avatar">🤖</div>
        <div class="typing-pill">
          <span class="dot"></span>
          <span class="dot"></span>
          <span class="dot"></span>
        </div>
      </div>
    </div>

    <div class="composer">
      <div class="input-shell">
        <textarea id="messageInput" placeholder="Message Gemini..."></textarea>
        <button id="sendButton" class="send-btn">Send</button>
      </div>
      <div class="footer-row">
        <div>Press Enter to send. Use Shift + Enter for a new line.</div>
        <div id="statusText">Ready</div>
      </div>
    </div>
  </div>

  <script>
    const chatBox = document.getElementById('chatBox');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const typingRow = document.getElementById('typingRow');
    const statusText = document.getElementById('statusText');

    function scrollToBottom() {
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    function setTyping(show) {
      typingRow.style.display = show ? 'flex' : 'none';
      statusText.textContent = show ? 'Gemini is thinking...' : 'Ready';
      scrollToBottom();
    }

    function addMessage(text, sender) {
      const row = document.createElement('div');
      row.className = `row ${sender}-row ${sender}`;

      const avatar = document.createElement('div');
      avatar.className = `avatar ${sender === 'user' ? 'user-avatar' : 'bot-avatar'}`;
      avatar.textContent = sender === 'user' ? '🙂' : '🤖';

      const wrap = document.createElement('div');
      wrap.className = 'bubble-wrap';

      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = sender === 'user' ? 'You' : 'Gemini';

      const bubble = document.createElement('div');
      bubble.className = 'message';
      bubble.textContent = text;

      wrap.appendChild(meta);
      wrap.appendChild(bubble);
      row.appendChild(avatar);
      row.appendChild(wrap);

      chatBox.insertBefore(row, typingRow);
      scrollToBottom();
    }

    async function sendMessage() {
      const text = messageInput.value.trim();
      if (!text) return;

      addMessage(text, 'user');
      messageInput.value = '';
      sendButton.disabled = true;
      messageInput.style.height = '58px';
      setTyping(true);

      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        });

        const data = await response.json();

        setTyping(false);

        if (!response.ok) {
          addMessage(`Error: ${data.error || 'Something went wrong.'}`, 'bot');
        } else {
          addMessage(data.reply, 'bot');
        }
      } catch (error) {
        setTyping(false);
        addMessage('Error: Could not reach the server.', 'bot');
      } finally {
        sendButton.disabled = false;
        messageInput.focus();
      }
    }

    messageInput.addEventListener('input', () => {
      messageInput.style.height = '58px';
      messageInput.style.height = Math.min(messageInput.scrollHeight, 180) + 'px';
    });

    sendButton.addEventListener('click', sendMessage);

    messageInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    });

    messageInput.focus();
  </script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/chat", methods=["POST"])
def chat_route():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    try:
        response = chat.send_message(user_message)
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
