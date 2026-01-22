import json
from typing import List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.analysts import ANALYST_CONFIG


class MacroNewsPick(BaseModel):
    company: str = Field(description="Company or basket name, e.g., 'Lockheed Martin' or 'Critical Minerals Basket'")
    ticker: Optional[str] = Field(default=None, description="Primary ticker if applicable, e.g., 'LMT' or 'NVDA'")
    sector: str = Field(description="Sector or theme, e.g., 'Defense', 'AI', 'Infrastructure', 'Critical Minerals'")
    thesis: str = Field(description="One-sentence investment thesis")
    strengths: str = Field(description="2–3 bullet-like strengths, concise; separated by '; '")
    risks: str = Field(description="1–2 key risks, concise; separated by '; '")
    best_experts: List[str] = Field(description="2–3 expert agent names chosen from the allowed list only")


class MacroNewsPicksOutput(BaseModel):
    picks: List[MacroNewsPick] = Field(description="Exactly five picks best aligned to the macro news context")

EXPERT_LIST = [
    f"{cfg['display_name']} Agent"
    for key, cfg in ANALYST_CONFIG.items()
]


def macro_news_opportunities_agent(state: AgentState, agent_id: str = "macro_news_opportunities_agent"):
    """
    Turn a macro news conversation or summary into 5 actionable company picks with short analyses
    and the best-fit expert agents (from the provided list) to evaluate each pick.
    
    Inputs (state['data']):
      - macro_news_context (str): optional free-text macro news summary or chat transcript
      - fallback_date (str): optional, purely for labeling in reasoning
    
    Outputs:
      - message (HumanMessage): JSON with {"picks": [...]} using MacroNewsPicksOutput schema
      - state['data']['macro_news_picks']: same JSON payload for downstream use
    """
    data = state.get("data", {})
    macro_news_context = data.get("macro_news_context") or ""
    fallback_date = data.get("fallback_date") or ""

    progress.update_status(agent_id, None, "Synthesizing macro context")

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an investment strategist agent. Convert the given macro news context into 5 "
                    "actionable company picks with concise theses and risks. Always choose the most salient, "
                    "liquid, and representative names tied to the macro catalysts.\n\n"
                    "Rules:\n"
                    "- Return strict JSON only.\n"
                    "- Produce exactly five picks.\n"
                    "- For each pick, select 2–3 'best_experts' ONLY from this list:\n"
                    f"{EXPERT_LIST}\n"
                    "- Keep 'strengths' and 'risks' concise; use '; ' as a separator between fragments.\n"
                    "- Prefer primary tickers where applicable. For baskets, leave 'ticker' null.\n"
                ),
            ),
            (
                "human",
                (
                    "Context date (optional): {date}\n"
                    "Macro news context:\n{context}\n\n"
                    "Return JSON with exactly this shape:\n"
                    "{{\n"
                    '  "picks": [\n'
                    "    {{\n"
                    '      "company": "string",\n'
                    '      "ticker": "string|null",\n'
                    '      "sector": "string",\n'
                    '      "thesis": "string",\n'
                    '      "strengths": "point; point; point",\n'
                    '      "risks": "risk; risk",\n'
                    '      "best_experts": ["Name Agent","Name Agent"]\n'
                    "    }}\n"
                    "  ]\n"
                    "}}\n"
                ),
            ),
        ]
    )

    prompt = template.invoke({"context": macro_news_context, "date": fallback_date})

    progress.update_status(agent_id, None, "Generating picks")
    llm_out = call_llm(
        prompt=prompt,
        pydantic_model=MacroNewsPicksOutput,
        agent_name=agent_id,
        state=state,
    )

    result_json = json.dumps(llm_out.model_dump(), ensure_ascii=False)
    message = HumanMessage(content=result_json, name=agent_id)

    # Save to state for downstream use
    state.setdefault("data", {})
    state["data"]["macro_news_picks"] = llm_out.model_dump()

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(llm_out.model_dump(), "Macro News Opportunities Agent")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state.get("messages", []) + [message],
        "data": state["data"],
    }

