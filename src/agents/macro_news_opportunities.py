import json
from typing import List, Optional

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm
from src.utils.progress import progress


# ------------------------------
# Data models
# ------------------------------

class MacroNewsPick(BaseModel):
    company: str
    ticker: Optional[str] = None
    sector: str
    thesis: str
    strengths: str
    risks: str
    best_experts: List[str]
    confidence: Optional[int] = Field(default=5, ge=1, le=5)


class MacroNewsPicksOutput(BaseModel):
    picks: List[MacroNewsPick]


EXPERT_LIST = [
    "Aswath Damodaran Agent",
    "Ben Graham Agent",
    "Bill Ackman Agent",
    "Cathie Wood Agent",
    "Charlie Munger Agent",
    "Michael Burry Agent",
    "Mohnish Pabrai Agent",
    "Peter Lynch Agent",
    "Phil Fisher Agent",
    "Rakesh Jhunjhunwala Agent",
    "Stanley Druckenmiller Agent",
    "Warren Buffett Agent",
]


# ------------------------------
# Agent
# ------------------------------

def macro_news_opportunities_agent(state: AgentState, agent_id: str = "macro_news_opportunities_agent"):
    data = state.get("data", {})
    metadata = state.get("metadata", {}) or {}
    node_data_by_id = metadata.get("node_data_by_id", {}) or {}
    this_node_data = node_data_by_id.get(agent_id, {}) or {}

    macro_news_context = data.get("macro_news_context") or this_node_data.get("macro_news_context")
    fallback_date = data.get("fallback_date") or this_node_data.get("fallback_date") or ""

    # Provide a safe default to avoid runtime failures if context is absent/short
    if not macro_news_context or len(macro_news_context.strip()) < 50:
        macro_news_context = (
            "Macro themes: NATO and Arctic security focus; Davos diplomacy; Ukraine peace efforts; "
            "AI and automation push across industries; supply chain resilience; critical minerals and rare earths."
        )

    progress.update_status(agent_id, None, "Synthesizing macro context")

    # System prompt guides the LLM how to choose companies dynamically
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an investment strategist agent. Convert the given macro-news context into 5 "
                "actionable company picks with concise theses, strengths, risks, confidence, and best-fit expert agents.\n\n"
                "Rules:\n"
                "- Return strict JSON only.\n"
                "- Produce exactly five picks.\n"
                "- Prioritize companies and sectors mentioned in the macro-news context.\n"
                "- Only add companies if a macro theme is mentioned but no specific company is identified.\n"
                "- Assign 2–3 'best_experts' ONLY from this list:\n"
                f"{EXPERT_LIST}\n"
                "- Do not assign the same expert to more than three picks.\n"
                "- Use at least four distinct experts across all picks.\n"
                "- Keep 'strengths' and 'risks' concise; separate items with '; '.\n"
                "- Prefer primary tickers; for baskets leave 'ticker' null.\n"
            ),
        ),
        (
            "human",
            (
                "Context date (optional): {date}\n"
                "Macro news context:\n{context}\n"
                 "Return JSON with exactly this shape:\n"
                 '{{ "picks": [ {{ "company": "string", "ticker": "string|null", "sector": "string", '
                 '"thesis": "string", "strengths": "point; point; point", "risks": "risk; risk", '
                 '"best_experts": ["Name Agent","Name Agent"], "confidence": 1-5 }} ] }}'
            ),
        )
    ])

    prompt = template.invoke({"context": macro_news_context, "date": fallback_date})

    progress.update_status(agent_id, None, "Generating picks")
    llm_out = call_llm(
        prompt=prompt,
        pydantic_model=MacroNewsPicksOutput,
        agent_name=agent_id,
        state=state,
    )

    # Normalize experts
    normalized_picks: List[MacroNewsPick] = []
    for pick in (llm_out.picks or []):
        pick.best_experts = [e for e in (pick.best_experts or []) if e in EXPERT_LIST][:3]
        normalized_picks.append(pick)

    if len(normalized_picks) > 5:
        normalized_picks = normalized_picks[:5]

    llm_out.picks = normalized_picks
    result_json = json.dumps(llm_out.model_dump(), ensure_ascii=False)
    message = AIMessage(content=result_json, name=agent_id)

    state.setdefault("data", {})
    state["data"]["macro_news_picks"] = llm_out.model_dump()

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(
            {
                "assumptions": "Derived dynamically from macro-news context",
                "output": llm_out.model_dump()
            },
            "Macro News Opportunities Agent"
        )

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state.get("messages", []) + [message],
        "data": state["data"],
    }
