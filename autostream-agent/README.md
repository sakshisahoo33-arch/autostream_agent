# AutoStream — Social-to-Lead AI Agent

> Built for the **ServiceHive / Inflx** ML Intern assignment.

---

## Project Overview

A LangGraph-powered conversational AI agent that turns social media conversations into qualified business leads for **AutoStream** — a fictional SaaS platform offering automated video editing tools for content creators.

---

## Quickstart

### 1. Clone & enter the project

```bash
git clone https://github.com/<your-username>/autostream-agent.git
cd autostream-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

```bash
export GOOGLE_API_KEY=AIza...
```

Or create a `.env` file:

```
GOOGLE_API_KEY=AIza...
```

### 5. Run the agent

```bash
python main.py
```

---

## Example Conversation

```
You: Hi there!
Agent: Hey! Welcome to AutoStream — the fastest way to edit and publish videos...

You: What plans do you offer?
Agent: We have two plans:
  • Basic — $29/month: 10 videos/month, 720p resolution...
  • Pro   — $79/month: Unlimited videos, 4K, AI captions, 24/7 support...

You: That sounds great, I want to try the Pro plan for my YouTube channel!
Agent: Great, I'd love to get you started! Could I start with your full name?

You: Riya Sharma
Agent: Nice to meet you, Riya Sharma! What's your email address?

You: riya@example.com
Agent: Perfect! Last question — which creator platform do you primarily use?

You: YouTube
Lead captured successfully: Riya Sharma, riya@example.com, YouTube
Agent: 🎉 You're all set, Riya Sharma! Your account request has been logged (ID: AS-04321)...
```

---

## Architecture (~200 words)

### Why LangGraph?

LangGraph was chosen because it models the agent as an **explicit state machine** rather than a free-form ReAct loop. Each node in the graph has a single responsibility — intent classification, knowledge retrieval, lead collection, or tool execution — making the flow predictable, debuggable, and easy to extend. Unlike AutoGen's multi-agent conversation model (better suited for autonomous task decomposition), LangGraph's graph topology maps directly to the linear-but-branching nature of a sales conversation.

### State Management

A typed `AgentState` dict is threaded through every node. It carries:
- `messages` — the full conversation history (accumulated via `add_messages` reducer so LangGraph merges turns correctly)
- `intent` — the latest classified intent (`greeting | product_query | high_intent`)
- `lead_name`, `lead_email`, `lead_platform` — progressively filled as the user provides them
- `lead_captured` — a boolean gate that prevents the lead tool from firing twice

Because all state lives in this single dict and is passed explicitly between nodes, the agent retains memory across 5–6 turns without any external memory store.

### RAG Pipeline

The knowledge base (`knowledge_base/autostream_kb.json`) is loaded at startup and serialised into a structured text block that is injected into the system prompt of the `respond_rag` node. For this assignment scope a flat keyword approach is sufficient; for production, the KB would be embedded (e.g., OpenAI / Cohere embeddings) and stored in a vector DB (Pinecone, Chroma) for semantic retrieval.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp:

1. **Create a Meta Business App** at [developers.facebook.com](https://developers.facebook.com) and enable the WhatsApp Business API.

2. **Register a Webhook URL** — a public HTTPS endpoint (e.g., a FastAPI or Flask server) that Meta will POST incoming messages to. During setup you provide a `verify_token` that Meta sends in a GET request; your server must echo it back to confirm ownership.

3. **Message handler** — in your webhook handler:
   ```python
   @app.post("/webhook")
   async def whatsapp_webhook(payload: dict):
       sender   = payload["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
       text     = payload["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]
       # Load session state for `sender` from Redis / DynamoDB
       state    = load_state(sender)
       state["messages"].append(HumanMessage(content=text))
       state    = graph.invoke(state)
       save_state(sender, state)
       reply    = last_ai_message(state)
       send_whatsapp_message(sender, reply)   # via Meta Graph API
   ```

4. **Session persistence** — replace the in-memory `AgentState` with a Redis or DynamoDB store keyed by the sender's WhatsApp number so state persists across HTTP requests.

5. **Send replies** — use the Meta Graph API (`POST /v18.0/{phone_number_id}/messages`) with a Bearer token to send the agent's reply back to the user.

This approach is stateless on the server side (each webhook call is independent) and horizontally scalable.

---

## Project Structure

```
autostream-agent/
├── agent/
│   ├── __init__.py
│   ├── graph.py        # LangGraph state machine (core agent logic)
│   └── rag.py          # Knowledge base loader & retriever
├── knowledge_base/
│   └── autostream_kb.json   # Pricing, features, policies, FAQ
├── tools/
│   ├── __init__.py
│   └── lead_capture.py      # mock_lead_capture() tool
├── main.py             # CLI entry point
├── requirements.txt
└── README.md
```

---

## Evaluation Checklist

| Criterion | Implementation |
|-----------|---------------|
| Intent detection | `classify_intent` node via LLM prompt |
| RAG retrieval | JSON KB → system prompt injection |
| State management | `AgentState` TypedDict across all nodes |
| Tool calling | `mock_lead_capture()` triggered only when name + email + platform collected |
| Memory (5–6 turns) | `add_messages` reducer accumulates full history |
| Code clarity | Modular nodes, single responsibility per file |
| WhatsApp deployment | Documented in README (webhook + session store pattern) |

---

## License

MIT
