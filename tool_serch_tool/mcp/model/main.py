"""
Run Qwen2.5-1.5B-Instruct as OpenAI-compatible API Server
Uses FastAPI - Works on Windows
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

print(f"Model loaded on: {model.device}")

app = FastAPI(title="Qwen2.5-1.5B API")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9


class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]


@app.get("/v1/models")
def list_models():
    return {
        "data": [
            {
                "id": "Qwen/Qwen2.5-1.5B-Instruct",
                "object": "model",
                "context_length": tokenizer.model_max_length,  # better to take it from model not from tokenizer
            }
        ]
    }


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("chat-api")


from fastapi import Request

@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    logger.info(
        f"Incoming request | {request.method} {request.url.path} | body={body.decode(errors='ignore')}"
    )

    response = await call_next(request)

    logger.info(
        f"Response status | {request.method} {request.url.path} | status={response.status_code}"
    )

    return response


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest) -> ChatResponse:
    """OpenAI-compatible chat completions endpoint"""
    
    # Convert messages to list of dicts
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response_text = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )
    
    return ChatResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=response_text),
                finish_reason="stop"
            )
        ]
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": model_name}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)