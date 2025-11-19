from fastapi import FastAPI, Query

from .schemas import AnswerResponse, HealthResponse
from .retriever import get_store, init_store
from .openai_agent import OpenAIQAAgent
from .semantic_only_agent import SemanticOnlyAgent

app = FastAPI()


@app.on_event("startup")
async def on_startup() -> None:
    # This runs INSIDE the uvicorn server process
    await init_store()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ask", response_model=AnswerResponse)
async def ask(
    q: str = Query(..., alias="q"),
    backend: str = Query("openai"),
) -> AnswerResponse:

    store = await get_store()
    results = store.semantic_search(q)

    # SEMANTIC BACKEND
    if backend == "semantic":
        agent = SemanticOnlyAgent()
        answer_text, reasoning_dict = await agent.answer(results)
        return AnswerResponse(
            answer=answer_text,
            reasoning=reasoning_dict,
        )

    # OPENAI BACKEND
    agent = OpenAIQAAgent()
    answer_text, reasoning_dict = agent.answer(q, results)
    return AnswerResponse(
        answer=answer_text,
        reasoning=reasoning_dict,
    )
