import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

app = FastAPI()

ALLOWED_ORIGINS = [
    "https://ben-gzf.github.io",
    "http://localhost:3000",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

class Msg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    messages: List[Msg] = []
    kb: Optional[str] = None 

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
async def chat(req: ChatReq):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return {"answer": "Server misconfigured: DEEPSEEK_API_KEY is missing on Render env vars."}
    last_user = ""
    for m in reversed(req.messages):
        if m.role == "user":
            last_user = m.content
            break

    kb = (req.kb or "").strip()

    system_prompt = """
You are BenBot, the personal website assistant for Zhefan (Ben) Guo.

Hard rules:
- Use ONLY facts that appear in the Knowledge Base.
- If the Knowledge Base does not contain the answer, say you don't have that information.
- Do NOT invent names, labs, achievements, dates, or links.
- If not in KB: say you don't have that information, and optionally suggest checking the resume.
- Do NOT use Markdown links. If sharing a link, output the raw URL only.
- Do NOT use any Markdown formatting (no **bold**, no backticks, no markdown bullets).
- When sharing any link, output the raw URL only (no surrounding punctuation like ** or parentheses).

Style:
- Sound friendly and human, not like a template.
- Vary phrasing (avoid repeating the same sentence patterns).
- Prefer short paragraphs and bullet points when helpful.
- You may add light conversational phrases as long as you do NOT add new facts.
- Make sure you act like a vivid chatAgent
- You can also include some Emojis, but not too often.
- If the user asks "who are you" or "who is Ben", give a 2-4 sentence intro, then offer what they can ask next.

Language: English.
""".strip()

    style_examples = """
Examples (style only):
User: what's my email?
Assistant: Sure — Ben's email is zhefan.guo@uconn.edu.

User: tell me who you are
Assistant: I'm BenBot — a small assistant on Ben's website. I can help with Ben's background, projects, and links (based only on what's listed on the site).

User: tell me who is Ben
Assistant: Ben (Zhefan Guo) is a Computer Science major with a Mathematics minor at UConn, graduating in 2026. Want his projects or links?
""".strip()

    payload: Dict[str, Any] = {
        "model": "deepseek-chat",
        "temperature": 0.5,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": style_examples},
            {"role": "system", "content": f"Knowledge Base:\n{kb}"},
            {"role": "user", "content": last_user},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(DEEPSEEK_API_URL, headers=headers, json=payload)
            data = r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if r.status_code >= 400:
        msg = (
            (data.get("error") or {}).get("message")
            or data.get("message")
            or str(data)
        )
        if "insufficient balance" in str(msg).lower():
            return {
                "answer": "BenBot is temporarily unavailable (API balance is not set up yet). Please check back later or use the contact email: zhefan.guo@uconn.edu."
            }
        return {"answer": f"DeepSeek API error: {msg}"}

    answer = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or "Sorry — no response."
    return {"answer": answer}
