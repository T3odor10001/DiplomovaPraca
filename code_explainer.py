from typing import List, Optional

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

EXPLAINER_PROMPT = """
You are a senior software engineer and code tutor.

You are given a USER QUESTION and a CODE CONTEXT extracted
from a real Python repository.

TASK:
Explain the code clearly and accurately.

RULES:
- Base your answer ONLY on the provided code context
- If something is not visible in the context, say so
- Refer to concrete files, classes, and methods
- Do NOT invent behavior
- Prefer step-by-step explanations
- If there is previous conversation, use it to understand follow-up questions
  (e.g. "that function" refers to something discussed earlier)

PREVIOUS CONVERSATION:
{history}

USER QUESTION:
{question}

CODE CONTEXT:
{context}
"""

def _format_history(conversation_history: Optional[List[dict]], max_entries: int = 4) -> str:
    """Format conversation history for the explainer prompt."""
    if not conversation_history:
        return "(none)"

    recent = conversation_history[-max_entries:]
    parts = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant" and len(content) > 500:
            content = content[:500] + "..."
        parts.append(f"{role.upper()}: {content}")

    return "\n".join(parts) if parts else "(none)"

def explain_code(
    question: str,
    context: str,
    model_name: str = "llama3.2",
    conversation_history: Optional[List[dict]] = None,
) -> str:
    model = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template(EXPLAINER_PROMPT)
    chain = prompt | model

    history_text = _format_history(conversation_history)

    result = chain.invoke(
        {
            "question": question,
            "context": context,
            "history": history_text,
        }
    )

    return str(result).strip()
