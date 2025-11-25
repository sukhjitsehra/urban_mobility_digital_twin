import ollama
from langchain.prompts import ChatPromptTemplate

MODEL = "llama3.1"
MAX_TURNS = 3

PROMPT_TEMPLATES = {
    "operator_a": """
You lead Operator A's micromobility network.
Demand hotspots: {demand_hotspots}
Ridership logs: {ridership_logs}
Accessibility requirements: {accessibility_requirements}
Regulator focus: {regulator_focus}

Conversation history:
{history}

Describe where you will expand or retreat coverage and the data justifying it.
""",
    "operator_b": """
You lead Operator B.
Demand hotspots: {demand_hotspots}
Ridership logs: {ridership_logs}
Accessibility requirements: {accessibility_requirements}
Regulator focus: {regulator_focus}

Conversation history:
{history}

Respond to Operator A, propose complementary zones, and share rebalancing plans.
""",
    "regulator": """
You are the regulator ensuring accessibility compliance.

Conversation history:
{history}

Evaluate the operators' plans, call out violations, and set reporting expectations.
""",
    "decider": """
You are the decider reviewing the micromobility negotiation.
Conversation so far:
{history}

If there is a joint plan that meets demand and accessibility needs, reply:
RESOLVED: <short note>
Otherwise reply:
CONTINUE: <blocking issue>
""",
}

AGENT_SEQUENCE = ["operator_a", "operator_b", "regulator"]

EXAMPLE_INPUTS = {
    "demand_hotspots": "Yeouido brunch corridor, Gangnam night riders, Mapo campus connectors",
    "ridership_logs": "Avg wait 6.5 min evenings, scooters idle 28% on weekdays",
    "accessibility_requirements": "Ramps within 50m of stations, audible alerts in CBD",
    "regulator_focus": "Prefer voluntary data sharing, warnings before fines this quarter",
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


def run_competition(
    demand_hotspots: str,
    ridership_logs: str,
    accessibility_requirements: str,
    regulator_focus: str,
    max_turns: int = MAX_TURNS,
):
    history = []
    turns = 0
    resolved = False
    shared_inputs = {
        "demand_hotspots": demand_hotspots,
        "ridership_logs": ridership_logs,
        "accessibility_requirements": accessibility_requirements,
        "regulator_focus": regulator_focus,
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
    run_competition(**EXAMPLE_INPUTS)
