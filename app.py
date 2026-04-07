import os
import ast
import operator as op
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from google import genai

app = Flask(__name__)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        'GEMINI_API_KEY is not set. In Render, add it as an environment variable.'
    )

client = genai.Client(api_key=API_KEY)
chat = client.chats.create(model="gemini-2.5-flash")

# -----------------------------
# Tiny beginner agent tools
# -----------------------------

ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
}


def safe_eval(expr: str):
    """Safely evaluate basic math like 2+2, (5*3)-1, 2**8."""

    def _eval(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numbers are allowed.")
        if isinstance(node, ast.BinOp):
            if type(node.op) not in ALLOWED_OPERATORS:
                raise ValueError("That math operator is not allowed.")
            return ALLOWED_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in ALLOWED_OPERATORS:
                raise ValueError("That unary operator is not allowed.")
            return ALLOWED_OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError("Invalid math expression.")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)


def run_tool(tool_name: str, tool_input: str):
    if tool_name == "time":
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Current server time: {now}"

    if tool_name == "calculator":
        result = safe_eval(tool_input)
        return f"Calculator result: {result}"

    return f"Unknown tool: {tool_name}"


def agent_reply(user_message: str) -> str:
    """
    A tiny agent loop:
    1. Ask Gemini whether to answer directly or use a tool
    2. If tool needed, run it in Python
    3. Ask Gemini to produce the final answer
    """
    planner_prompt = f"""
You are a tiny beginner AI agent.

You have 2 tools:
1. time -> use for current date/time questions
2. calculator -> use for arithmetic like 2+2, 15*7, (8+4)/2

Decide the NEXT step only.

Reply in EXACTLY ONE of these formats:

REPLY: <your direct reply>
TOOL: <tool_name> | <tool_input>

Rules:
- Use time only for time/date questions.
- Use calculator only for arithmetic.
- For anything else, use REPLY.
- Do not explain the format.

User message: {user_message}
""".strip()

    plan_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=planner_prompt,
    )

    plan_text = (plan_response.text or "").strip()

    if plan_text.startswith("TOOL:"):
        try:
            tool_part = plan_text[len("TOOL:"):].strip()
            tool_name, tool_input = [x.strip() for x in tool_part.split("|", 1)]
            tool_result = run_tool(tool_name, tool_input)
        except Exception as e:
            return f"I tried to use a tool, but something went wrong: {e}"

        final_prompt = f"""
You are a helpful assistant.
The user asked: {user_message}
You used the tool '{tool_name}'.
Tool result: {tool_result}

Now answer the user naturally and clearly.
""".strip()

        final_response = chat.send_message(final_prompt)
        return final_response.text

    if plan_text.startswith("REPLY:"):
        return plan_text[len("REPLY:"):].strip()

    # Fallback if Gemini does not follow the format exactly
    fallback_response = chat.send_message(user_message)
    return fallback_response.text


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>My Gemini Agent</title>
  <style>
    :root {
      --bg-1: #020617;
      --bg-2: #0f172a;
      --panel: rgba(15, 23, 42, 0.82);
      --panel-strong: rgba(15, 23, 42, 0.96);
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
      background: linear-gradient(180deg, rgba(2, 6, 23, 0.18), rgba(2, 6, 23, 0.32));
    }

    .row {
      display: flex;
      align-items: flex-end;
      gap: 12px;
      animation: fadeInUp 0.22s ease;
    }

    .row.user-row { justify-content: flex-end; }

    .avatar {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      flex-shrink: 0;
      font-size: 16px;
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

    .send-btn {
      min-width: 112px;
      border: none;
      border-radius: 18px;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
      color: white;
      font-weight: 700;
      font-size: 15px;
      cursor: pointer;
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
  </style>
</head>
<body>
  <div class="app">
    <div class="header">
      <div class="brand">
        <div class="logo">✦</div>
        <div class="title-wrap">
          <h1>My Gemini Agent</h1>
          <p>My first tiny AI agent with tools.</p>
        </div>
      </div>
      <div class="badge">Tools: time + calculator</div>
    </div>

    <div id="chatBox" class="chat-box">
      <div class="row bot-row bot">
        <div class="avatar bot-avatar">🤖</div>
        <div class="bubble-wrap">
          <div class="meta">Agent</div>
          <div class="message">Hi! I am your first tiny AI agent. I can chat, do math, and check the server time.</div>
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
        <textarea id="messageInput" placeholder="Try: what time is it? or calculate (12*8)+5"></textarea>
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
      statusText.textContent = show ? 'Agent is thinking...' : 'Ready';
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
      meta.textContent = sender === 'user' ? 'You' : 'Agent';

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
        reply = agent_reply(user_message)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
