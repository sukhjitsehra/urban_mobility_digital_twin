import ollama
from langchain_core.prompts import ChatPromptTemplate

MODEL = "llama3.1"
MAX_TURNS = 3

PROMPT_TEMPLATES = {
    "fleet": """
You manage the fleet.
Emissions goal: {emissions_goal}
Vehicle usage: {vehicle_usage}
Energy constraints: {energy_constraints}
Compliance notes: {compliance_notes}

Conversation history:
{history}

Propose operational moves (dispatch tweaks, vehicle swaps) to hit the emissions goal.
""",
    "energy": """
You represent the energy utility.

Conversation history:
{history}

Ensure fleet plans fit {energy_constraints} and suggest charging windows or infrastructure.
""",
    "watchdog": """
You are the environmental watchdog referencing simul_result outputs.

Conversation history:
{history}

Check compliance vs. {compliance_notes} and request monitoring artifacts.
""",
    "decider": """
You are the decider overseeing the environmental caucus.
Conversation so far:
{history}

If a coordinated plan meets {emissions_goal}, respond:
RESOLVED: <summary>
Otherwise respond:
CONTINUE: <remaining issue>
""",
}

AGENT_SEQUENCE = ["fleet", "energy", "watchdog"]

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


def run_environmental_board(
    emissions_goal: str,
    vehicle_usage: str,
    energy_constraints: str,
    compliance_notes: str,
    max_turns: int = MAX_TURNS,
):
    history = []
    turns = 0
    resolved = False
    shared_inputs = {
        "emissions_goal": emissions_goal,
        "vehicle_usage": vehicle_usage,
        "energy_constraints": energy_constraints,
        "compliance_notes": compliance_notes,
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


## Example run
if __name__ == "__main__":
    run_environmental_board(
        emissions_goal="Cut CO2e by 12% this quarter while keeping service levels",
        vehicle_usage="38% of trips still diesel vans; EV fleet idle midday",
        energy_constraints="Grid prefers charging after 22:00; max 4 fast chargers in Gangnam",
        compliance_notes="Watchdog flagged missing telemetry for 5 vehicles in last audit",
    )
