import ollama
from langchain.prompts import ChatPromptTemplate

MODEL = "llama3.1"
MAX_TURNS = 3

PROMPT_TEMPLATES = {
    "marketing": """
You are the marketing lead.
Pricing goal: {pricing_goal}
Marketing storyline: {marketing_focus}
Operations limits: {operations_limits}
Dashboard needs: {dashboard_snapshot}
Citizen feedback: {citizen_feedback}

Conversation history:
{history}

Pitch a pricing narrative and surface the KPIs dashboard.py should highlight.
""",
    "operations": """
You are the operations lead with simulator_helper.py constraints.

Conversation history:
{history}

Stress-test the marketing idea against {operations_limits} and explain needed guardrails.
""",
    "citizen": """
You speak for citizen advisors.

Conversation history:
{history}

Judge the proposal against {citizen_feedback} and note transparency requirements for dashboard.py.
""",
    "decider": """
You are the decider moderating the pricing debate.
Conversation so far:
{history}

If the parties converged on an experiment backlog that satisfies {pricing_goal}, answer:
RESOLVED: <summary>
Otherwise answer:
CONTINUE: <gap>
""",
}

AGENT_SEQUENCE = ["marketing", "operations", "citizen"]

EXAMPLE_INPUTS = {
    "pricing_goal": "Hit +5% revenue without exceeding 9 min average wait",
    "marketing_focus": "Frame shared rides as climate-friendly while keeping VIP tier",
    "operations_limits": "Fleet maxes out at 120 concurrent trips; surge >1.6x triggers cancellations",
    "dashboard_snapshot": "Need tiles showing wait time distribution and revenue bands",
    "citizen_feedback": "Worried about price spikes for medical trips; want advance notice",
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


def run_pricing_table(
    pricing_goal: str,
    marketing_focus: str,
    operations_limits: str,
    dashboard_snapshot: str,
    citizen_feedback: str,
    max_turns: int = MAX_TURNS,
):
    history = []
    turns = 0
    resolved = False
    shared_inputs = {
        "pricing_goal": pricing_goal,
        "marketing_focus": marketing_focus,
        "operations_limits": operations_limits,
        "dashboard_snapshot": dashboard_snapshot,
        "citizen_feedback": citizen_feedback,
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
    run_pricing_table(**EXAMPLE_INPUTS)
