"""
agent/graph.py
LangGraph-based agentic workflow for AutoStream social-to-lead conversion.
"""

import os
from typing import Annotated, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agent.rag import retrieve_context
from tools.lead_capture import mock_lead_capture

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    lead_name: str | None
    lead_email: str | None
    lead_platform: str | None
    lead_captured: bool
    collecting_lead: bool   # flag: we are in lead-collection mode

# ---------------------------------------------------------------------------
# LLM setup (Groq — free)
# ---------------------------------------------------------------------------

def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=512,
    )

# ---------------------------------------------------------------------------
# Shared KB context
# ---------------------------------------------------------------------------

KB_CONTEXT = retrieve_context("")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def last_user_msg(state: AgentState) -> str:
    return next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    ).strip()

# ---------------------------------------------------------------------------
# Node 1 — Classify intent
# ---------------------------------------------------------------------------

INTENT_SYSTEM = """
You are an intent classifier for AutoStream, a SaaS video editing tool.
Classify the LAST user message into EXACTLY one of:
  - greeting       (hello, hi, hey, thanks, bye, small talk)
  - product_query  (questions about features, pricing, plans, policies)
  - high_intent    (ready to sign up, want to try, yes let's go, purchase intent)

Reply with ONLY the label — no punctuation, no explanation.
""".strip()

def classify_intent(state: AgentState) -> AgentState:
    # If already collecting lead info, skip classification
    if state.get("collecting_lead"):
        return {**state, "intent": "high_intent"}

    llm = get_llm()
    user_msg = last_user_msg(state)
    result = llm.invoke([
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    intent = result.content.strip().lower()
    if intent not in ("greeting", "product_query", "high_intent"):
        intent = "product_query"
    return {**state, "intent": intent}

# ---------------------------------------------------------------------------
# Node 2a — Greeting
# ---------------------------------------------------------------------------

GREETING_SYSTEM = f"""
You are a friendly sales assistant for AutoStream — automated video editing for creators.
Keep replies to 2-3 sentences. Be warm, invite the user to ask about our product.

Company context:
{KB_CONTEXT}
""".strip()

def respond_general(state: AgentState) -> AgentState:
    llm = get_llm()
    reply = llm.invoke([SystemMessage(content=GREETING_SYSTEM)] + state["messages"])
    return {**state, "messages": state["messages"] + [AIMessage(content=reply.content)]}

# ---------------------------------------------------------------------------
# Node 2b — RAG product/pricing response
# ---------------------------------------------------------------------------

RAG_SYSTEM = f"""
You are a knowledgeable product advisor for AutoStream.
Answer the user's question accurately using ONLY the information below.
Be concise but complete. Do NOT ask for personal details or offer to sign anyone up.

--- KNOWLEDGE BASE ---
{KB_CONTEXT}
--- END ---
""".strip()

def respond_rag(state: AgentState) -> AgentState:
    llm = get_llm()
    reply = llm.invoke([SystemMessage(content=RAG_SYSTEM)] + state["messages"])
    return {**state, "messages": state["messages"] + [AIMessage(content=reply.content)]}

# ---------------------------------------------------------------------------
# Node 2c — Hardcoded lead collection (step-by-step, no LLM hallucination)
# ---------------------------------------------------------------------------

def collect_lead(state: AgentState) -> AgentState:
    new_state = dict(state)
    user_input = last_user_msg(state)
    already_collecting = state.get("collecting_lead", False)
    new_state["collecting_lead"] = True

    name = state.get("lead_name")

    if not already_collecting:
        # First time — user just said "I want to sign up". Ask for name.
        reply = (
            "Great choice! I'll get you signed up for AutoStream right now.\n\n"
            "What's your **full name**?"
        )
    elif name is None:
        # User just typed their name
        new_state["lead_name"] = user_input
        reply = f"Nice to meet you, **{user_input}**! 👋\n\nWhat's your **email address**?"
    else:
        reply = "Just a moment…"

    new_state["messages"] = state["messages"] + [AIMessage(content=reply)]
    return new_state

# ---------------------------------------------------------------------------
# Node 2d — Ask for email → platform, then capture
# ---------------------------------------------------------------------------

def collect_email(state: AgentState) -> AgentState:
    new_state = dict(state)
    user_input = last_user_msg(state)

    # Validate email
    if "@" not in user_input or "." not in user_input.split("@")[-1]:
        reply = "That doesn't look like a valid email address. Could you re-enter it? (e.g. name@gmail.com)"
    else:
        new_state["lead_email"] = user_input
        reply = (
            f"Got it — **{user_input}** ✅\n\n"
            "Last question: which **creator platform** do you mainly use?\n"
            "(e.g. YouTube, Instagram, TikTok, Twitch…)"
        )

    new_state["messages"] = state["messages"] + [AIMessage(content=reply)]
    return new_state

# ---------------------------------------------------------------------------
# Node 2e — Fire mock_lead_capture()
# ---------------------------------------------------------------------------

def capture_lead(state: AgentState) -> AgentState:
    new_state = dict(state)
    platform = last_user_msg(state)
    new_state["lead_platform"] = platform

    result = mock_lead_capture(
        name=new_state["lead_name"],
        email=new_state["lead_email"],
        platform=platform,
    )

    if result["status"] == "success":
        lead = result["lead"]
        reply = (
            f"🎉 Welcome aboard, **{lead['name']}**!\n\n"
            f"Your AutoStream account has been created (ID: `{lead['lead_id']}`).\n"
            f"We'll send your login details and free trial info to **{lead['email']}** shortly.\n\n"
            f"Enjoy editing on **{platform}**! 🚀"
        )
        new_state["lead_captured"] = True
        new_state["collecting_lead"] = False
    else:
        reply = f"Something went wrong: {result['message']} — please try again."

    new_state["messages"] = state["messages"] + [AIMessage(content=reply)]
    return new_state

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_classify(state: AgentState) -> str:
    if state.get("lead_captured"):
        return "end"
    intent = state.get("intent", "product_query")
    collecting = state.get("collecting_lead", False)
    name = state.get("lead_name")
    email = state.get("lead_email")

    if collecting or intent == "high_intent":
        if name is None:
            return "collect_lead"
        elif email is None:
            return "collect_email"
        else:
            return "capture_lead"

    if intent == "greeting":
        return "respond_general"
    return "respond_rag"

# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("classify_intent", classify_intent)
    g.add_node("respond_general", respond_general)
    g.add_node("respond_rag", respond_rag)
    g.add_node("collect_lead", collect_lead)
    g.add_node("collect_email", collect_email)
    g.add_node("capture_lead", capture_lead)

    g.set_entry_point("classify_intent")

    g.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "respond_general": "respond_general",
            "respond_rag": "respond_rag",
            "collect_lead": "collect_lead",
            "collect_email": "collect_email",
            "capture_lead": "capture_lead",
            "end": END,
        },
    )

    g.add_edge("respond_general", END)
    g.add_edge("respond_rag", END)
    g.add_edge("collect_lead", END)
    g.add_edge("collect_email", END)
    g.add_edge("capture_lead", END)

    return g.compile()
