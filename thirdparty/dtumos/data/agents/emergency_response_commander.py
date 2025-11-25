import ollama
from langchain_core.prompts import ChatPromptTemplate

MODEL = "llama3.1"
MAX_TURNS = 3

PROMPT_TEMPLATES = {
    "commander": """
You are the emergency commander.
Incident: {incident_report}
Road status: {road_status}
Resources: {resource_status}
Support requests: {support_requests}
Mayor priority: {mayor_priority}

Conversation history:
{history}

Lay out immediate stabilization moves and vehicle reroutes using module/osrm_routing.py data.
""",
    "hospital": """
You coordinate hospital + social services.

Conversation history:
{history}

Allocate beds, social workers, and vans based on {resource_status} and requests {support_requests}.
""",
    "mayor_staff": """
You brief the mayor dashboard.

Conversation history:
{history}

Summarize KPIs tied to {mayor_priority} and flag risks or comms needs for passengers.
""",
    "decider": """
You are the decider checking if the emergency response is aligned.
Conversation so far:
{history}

If the plan addresses reroutes, resources, and communications, reply:
RESOLVED: <brief>
Otherwise reply:
CONTINUE: <gap>
""",
}

AGENT_SEQUENCE = ["commander", "hospital", "mayor_staff"]

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


def run_incident_room(
    incident_report: str,
    road_status: str,
    resource_status: str,
    support_requests: str,
    mayor_priority: str,
    max_turns: int = MAX_TURNS,
):
    history = []
    turns = 0
    resolved = False
    shared_inputs = {
        "incident_report": incident_report,
        "road_status": road_status,
        "resource_status": resource_status,
        "support_requests": support_requests,
        "mayor_priority": mayor_priority,
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
    run_incident_room(
        incident_report="Chemical spill near Jongno 3-ga at 17:10, 3 blocked intersections",
        road_status="OSRM shows closures on Jong-ro, detour adds 6 min via Insadong",
        resource_status="Hospitals A/B at 70% capacity, 8 social workers free, 15 accessible vans",
        support_requests="12 dialysis passengers stuck downtown needing reroute",
        mayor_priority="Protect vulnerable riders, reopen transit spine within 45 min",
    )
