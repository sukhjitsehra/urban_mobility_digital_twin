import ollama
from langchain.prompts import ChatPromptTemplate

MODEL = "llama3.1"
MAX_TURNS = 3

PROMPT_TEMPLATES = {
    "dispatcher": """
You are the dispatcher agent assigning vehicles using module/dispatch.py heuristics.
Passenger request: {trip_request}
Fleet snapshot: {fleet_state}
Routing analysis from module/osrm_routing.py: {routing_snapshot}
Negotiation target: {negotiation_goal}
Risk tolerance: {risk_tolerance}

Conversation history:
{history}

Respond with the best vehicle choice, ETA, and how it affects margins.
""",
    "finance": """
You are the finance agent ensuring profitability using dispatch_cost.py metrics.
Passenger request: {trip_request}
Fleet snapshot: {fleet_state}
Routing analysis: {routing_snapshot}
Negotiation target: {negotiation_goal}
Risk tolerance: {risk_tolerance}

Conversation history:
{history}

React to the dispatcher plan, adjust pricing or incentives, and explain rider messaging.
""",
    "decider": """
You are the decider evaluating the taxi negotiation thread.
Conversation so far:
{history}

If the team reached a viable plan that meets pickup and margin targets, reply:
RESOLVED: <one-sentence reason>
Otherwise reply:
CONTINUE: <what is missing>
""",
}

AGENT_SEQUENCE = ["dispatcher", "finance"]

EXAMPLE_INPUTS = {
    "trip_request": "ID 8821, wheelchair passenger, pickup in Mapo at 08:05 needing hospital drop-off",
    "fleet_state": "Vehicle T17 finishing trip in Yongsan, T04 deadheading near Mapo, fuel OK",
    "routing_snapshot": "OSRM fastest path 11 min via Hangang-daero; tolled alternative adds 3 min but higher fare",
    "negotiation_goal": "Keep pickup under 12 minutes while maintaining >=8% margin",
    "risk_tolerance": "balanced",
}


def run_agent(agent_name, **kwargs):
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATES[agent_name]).format(**kwargs)
    print(f"\n[{agent_name.upper()}]\n", end="")
    response_text = ""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in response:
        part = chunk["message"]["content"]
        response_text += part
        print(part, end="", flush=True)
    print()
    return response_text.strip()


def run_negotiation(
    trip_request: str,
    fleet_state: str,
    routing_snapshot: str,
    negotiation_goal: str,
    risk_tolerance: str = "balanced",
    max_turns: int = MAX_TURNS,
):
    history = []
    turns = 0
    resolved = False
    shared_inputs = {
        "trip_request": trip_request,
        "fleet_state": fleet_state,
        "routing_snapshot": routing_snapshot,
        "negotiation_goal": negotiation_goal,
        "risk_tolerance": risk_tolerance,
    }

    while turns < max_turns and not resolved:
        for agent in AGENT_SEQUENCE:
            message = run_agent(agent, history="\n".join(history), **shared_inputs)
            history.append(f"{agent}: {message}")
        decision = run_agent("decider", history="\n".join(history), **shared_inputs)
        history.append(f"decider: {decision}")
        if decision.upper().startswith("RESOLVED"):
            resolved = True
        turns += 1


if __name__ == "__main__":
    run_negotiation(**EXAMPLE_INPUTS)
