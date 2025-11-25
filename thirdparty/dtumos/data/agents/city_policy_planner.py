import ollama
from langchain.prompts import ChatPromptTemplate

MODEL = "llama3.1"
MAX_TURNS = 3

PROMPT_TEMPLATES = {
    "planner": """
You are the city planning agent testing policies through module/dispatch.py and simulator.py results.
Policy levers: {policy_levers}
Stakeholder positions: {stakeholder_positions}
Dispatch KPIs: {dispatch_metrics}
Simulation summary: {simulation_summary}
Budget guardrails: {budget_guardrails}

Conversation history:
{history}

Propose the next simulation-ready bundle, including expected KPI shifts.
""",
    "citizen": """
You represent citizen + accessibility coalitions reviewing dashboard.py evidence.
Policy levers: {policy_levers}
Stakeholder positions: {stakeholder_positions}
Dispatch KPIs: {dispatch_metrics}
Simulation summary: {simulation_summary}
Budget guardrails: {budget_guardrails}

Conversation history:
{history}

React to the planner proposal, flag equity issues, and add communication needs.
""",
    "budget": """
You are the budget analyst checking simulator_helper.py run costs.
Conversation history:
{history}

Ensure the plan stays within {budget_guardrails} and note any trade-offs for finance sign-off.
""",
    "decider": """
You are the decider. Conversation so far:
{history}

If consensus exists on a test matrix meeting KPI and budget goals, answer:
RESOLVED: <short reason>
Otherwise answer:
CONTINUE: <missing work>
""",
}

AGENT_SEQUENCE = ["planner", "citizen", "budget"]

EXAMPLE_INPUTS = {
    "policy_levers": "5k KRW off-peak subsidy, CBD congestion fee, expanded wheelchair service radius",
    "stakeholder_positions": "Planners favor congestion fee, citizen group wary of surcharges, budget wants net neutral",
    "dispatch_metrics": "Baseline wait 9.2 min, subsidy scenario 7.8 min, congestion fee raises revenue by 6%",
    "simulation_summary": "Scenario_042 shows 4% ridership gain but 2% budget shortfall",
    "budget_guardrails": "Annual pilot must stay within 1.5B KRW new spend",
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


def run_policy_lab(
    policy_levers: str,
    stakeholder_positions: str,
    dispatch_metrics: str,
    simulation_summary: str,
    budget_guardrails: str,
    max_turns: int = MAX_TURNS,
):
    history = []
    turns = 0
    resolved = False
    shared_inputs = {
        "policy_levers": policy_levers,
        "stakeholder_positions": stakeholder_positions,
        "dispatch_metrics": dispatch_metrics,
        "simulation_summary": simulation_summary,
        "budget_guardrails": budget_guardrails,
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
    run_policy_lab(**EXAMPLE_INPUTS)
