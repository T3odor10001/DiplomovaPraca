from __future__ import annotations

from typing import TypedDict, Optional, Dict, Any, Callable
from langgraph.graph import StateGraph, END

class DocReviewState(TypedDict, total=False):
    path: str
    code: str

    round: int
    max_rounds: int

    feedback_for_writer: str

    documentation: str
    review_text: str
    decision: Optional[str]
    feedback: str

def parse_decision(text: str):
    upper = text.upper()
    if "DECISION: REVISE" in upper:
        decision = "REVISE"
    elif "DECISION: APPROVE" in upper:
        decision = "APPROVE"
    else:
        decision = None

    feedback = ""
    if "FEEDBACK:" in upper:
        idx = upper.find("FEEDBACK:")
        feedback = text[idx + len("FEEDBACK:") :].strip()

    return decision, feedback

def build_doc_review_graph(docwriter, reviewer):
    """
    LangGraph workflow:
      write -> review -> (APPROVE => END) else -> write ... max_rounds
    docwriter/reviewer sú LangChain runnables (u teba prompt | model).
    """

    def write_node(state: DocReviewState) -> Dict[str, Any]:
        doc = docwriter.invoke(
            {
                "path": state["path"],
                "code": state["code"],
                "feedback": state.get("feedback_for_writer", ""),
            }
        )
        return {"documentation": str(doc)}

    def review_node(state: DocReviewState) -> Dict[str, Any]:
        review = reviewer.invoke(
            {
                "path": state["path"],
                "code": state["code"],
                "doc": state.get("documentation", ""),
            }
        )
        review_text = str(review)
        decision, feedback = parse_decision(review_text)

        updates: Dict[str, Any] = {
            "review_text": review_text,
            "decision": decision,
            "feedback": feedback,
        }

        if decision != "APPROVE":
            updates["feedback_for_writer"] = feedback or (
                "Please fix coverage, correctness, clarity, and parameter notes as needed."
            )
            updates["round"] = int(state.get("round", 0)) + 1

        return updates

    def should_continue(state: DocReviewState) -> str:
        if state.get("decision") == "APPROVE":
            return "end"

        if int(state.get("round", 0)) >= int(state.get("max_rounds", 2)):
            return "end"

        return "continue"

    g = StateGraph(DocReviewState)
    g.add_node("write", write_node)
    g.add_node("review", review_node)

    g.set_entry_point("write")
    g.add_edge("write", "review")
    g.add_conditional_edges("review", should_continue, {"continue": "write", "end": END})

    return g.compile()
