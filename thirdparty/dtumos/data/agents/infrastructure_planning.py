import ollama
from langchain_core.prompts import ChatPromptTemplate

MODEL = "llama3.1"
MAX_TURNS = 3

PROMPT_TEMPLATES = {
    "designer": """
You are the urban designer proposing physical interventions.
Travel patterns: {travel_patterns}
OSRM obstacles: {osrm_obstacles}
Budget notes: {budget_notes}
Equity requests: {equity_requests}

Conversation history:
{history}

Recommend sites, expected travel-time gains, and construction sequencing.
""",
    "advocate": """
You represent accessibility advocates.

Conversation history:
{history}

Assess whether the designer plan satisfies {equity_requests} and detail missing accommodations.
""",
    "developer": """
You are the developer/budget reviewer focusing on {budget_notes}.

Conversation history:
{history}

Discuss funding structure, phasing, and risks tied to OSRM constraints {osrm_obstacles}.
""",
    "decider": """
You are the decider summarizing the infrastructure charrette.
Conversation so far:
{history}

If a consensus package meets access and budget needs, reply:
RESOLVED: <short reason>
Otherwise reply:
CONTINUE: <blocking topic>
""",
}

AGENT_SEQUENCE = ["designer", "advocate", "developer"]

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


def run_charrette(
    travel_patterns: str,
    osrm_obstacles: str,
    budget_notes: str,
    equity_requests: str,
    max_turns: int = MAX_TURNS,
):
    history = []
    turns = 0
    resolved = False
    shared_inputs = {
        "travel_patterns": travel_patterns,
        "osrm_obstacles": osrm_obstacles,
        "budget_notes": budget_notes,
        "equity_requests": equity_requests,
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
    run_charrette(
        travel_patterns="East-west commuter surge mornings; nighttime trips clustering around hospital",
        osrm_obstacles="Steep ramps near Itaewon, limited curb space in Jongno",
        budget_notes="Capex 12B KRW over 2 years, prefer public-private mix",
        equity_requests="Prioritize wheelchair ramps near clinics and add EV chargers near social housing",
    )
