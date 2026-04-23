"""
main.py
CLI entry point for the AutoStream Social-to-Lead agent.

Run:
    python main.py

Environment variables required:
    GROQ_API_KEY=<your key>
"""

import os
import sys

from langchain_core.messages import HumanMessage

from agent.graph import AgentState, build_graph

# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if not os.getenv("GROQ_API_KEY"):
    sys.exit(
        "ERROR: GROQ_API_KEY environment variable is not set.\n"
        "Export it before running:  export GROQ_API_KEY=gsk_..."
    )

# ---------------------------------------------------------------------------
# Build graph (once)
# ---------------------------------------------------------------------------

graph = build_graph()

# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

state: AgentState = {
    "messages": [],
    "intent": "",
    "lead_name": None,
    "lead_email": None,
    "lead_platform": None,
    "lead_captured": False,
    "collecting_lead": False,
}

# ---------------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------------

BANNER = """
╔══════════════════════════════════════════════════╗
║   AutoStream  ·  AI Sales Agent  (Inflx/ServiceHive) ║
╚══════════════════════════════════════════════════╝
Type your message and press Enter.  Ctrl+C to quit.
"""

print(BANNER)

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break

    if not user_input:
        continue

    # Append user message to state
    state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

    # Run graph
    try:
        state = graph.invoke(state)
    except Exception as exc:
        print(f"[Agent Error] {exc}")
        continue

    # Print last AI message
    from langchain_core.messages import AIMessage
    ai_msgs = [m for m in state["messages"] if isinstance(m, AIMessage)]
    if ai_msgs:
        print(f"\nAgent: {ai_msgs[-1].content}\n")

    # End conversation after successful lead capture
    if state.get("lead_captured"):
        print("─── Lead captured. Session complete. ───")
        break
