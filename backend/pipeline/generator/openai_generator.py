from openai import OpenAI
from .base import BaseGenerator, GeneratorResponse, Message

MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = """당신은 RFP(제안요청서) 문서 분석 전문가입니다.
주어진 컨텍스트만을 기반으로 질문에 답변하세요.
컨텍스트에 없는 내용은 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요.
답변은 한국어로 작성하세요."""


def _build_context_block(chunks: list[str]) -> str:
    sections = "\n\n---\n\n".join(f"[문서 {i+1}]\n{chunk}" for i, chunk in enumerate(chunks))
    return f"## 참조 문서\n\n{sections}"


class OpenAIGenerator(BaseGenerator):
    def __init__(self, api_key: str, model: str = MODEL):
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(
        self,
        query: str,
        context_chunks: list[str],
        history: list[Message] | None = None,
    ) -> GeneratorResponse:
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})

        context_block = _build_context_block(context_chunks)
        user_content = f"{context_block}\n\n## 질문\n\n{query}"
        messages.append({"role": "user", "content": user_content})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()

        new_history = list(history) if history else []
        new_history.append(Message(role="user", content=query))
        new_history.append(Message(role="assistant", content=answer))

        return GeneratorResponse(answer=answer, history=new_history)
