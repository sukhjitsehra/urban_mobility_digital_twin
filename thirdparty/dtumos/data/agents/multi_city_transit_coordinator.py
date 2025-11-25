import ollama
from langchain.prompts import ChatPromptTemplate

MODEL = "llama3.1"
MAX_TURNS = 3

PROMPT_TEMPLATES = {
    "logistics": """
You are the logistics operator.
City pairing: {city_pairing}
Transfer metrics: {transfer_metrics}
Integration constraints: {integration_constraints}
Payment expectations: {payment_expectations}

Conversation history:
{history}

Explain freight and rideshare needs plus stop adjustments to cut missed connections.
""",
    "transit": """
You represent the public transit agency.

Conversation history:
{history}

Address {integration_constraints}, propose scheduling/API fixes, and note rider experience impacts.
""",
    "rideshare": """
You speak for rideshare/payment partnerships.

Conversation history:
{history}

React to the combined plan, highlight payment negotiation red lines, and KPIs for visualization/dashboard/.
""",
    "decider": """
You are the decider moderating multi-city coordination.
Conversation so far:
{history}

If there is a partnership framework hitting reliability + payment needs, respond:
RESOLVED: <summary>
Otherwise respond:
CONTINUE: <blocker>
""",
}

AGENT_SEQUENCE = ["logistics", "transit", "rideshare"]

EXAMPLE_INPUTS = {
    "city_pairing": "Seoul-Incheon airport corridor with feeder buses",
    "transfer_metrics": "Current transfer wait 14 min avg; missed connections 11% during storms",
    "integration_constraints": "Metro API rate limits, limited curb access at Terminal 1",
    "payment_expectations": "Logistics wants minimum revenue floor; transit agency insists on tap-to-pay parity",
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


def run_coordination(
    city_pairing: str,
    transfer_metrics: str,
    integration_constraints: str,
    payment_expectations: str,
    max_turns: int = MAX_TURNS,
):
    history = []
    turns = 0
    resolved = False
    shared_inputs = {
        "city_pairing": city_pairing,
        "transfer_metrics": transfer_metrics,
        "integration_constraints": integration_constraints,
        "payment_expectations": payment_expectations,
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
    run_coordination(**EXAMPLE_INPUTS)
