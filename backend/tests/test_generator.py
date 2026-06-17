import pytest
from unittest.mock import patch, MagicMock
from pipeline.generator.openai_generator import OpenAIGenerator
from pipeline.generator.base import Message, GeneratorResponse


def _make_mock_client(answer: str) -> MagicMock:
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=answer))]
    )
    return mock_client


@pytest.fixture
def mock_openai():
    with patch("pipeline.generator.openai_generator.OpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_client("테스트 답변입니다.")
        yield mock_cls


def test_generator_returns_response(mock_openai):
    gen = OpenAIGenerator(api_key="test-key")
    result = gen.generate(query="사업 예산은?", context_chunks=["예산은 1억원입니다."])

    assert isinstance(result, GeneratorResponse)
    assert result.answer == "테스트 답변입니다."


def test_generator_history_appended(mock_openai):
    gen = OpenAIGenerator(api_key="test-key")
    result = gen.generate(query="사업 예산은?", context_chunks=["예산은 1억원입니다."])

    assert len(result.history) == 2
    assert result.history[0].role == "user"
    assert result.history[0].content == "사업 예산은?"
    assert result.history[1].role == "assistant"
    assert result.history[1].content == "테스트 답변입니다."


def test_generator_carries_history(mock_openai):
    gen = OpenAIGenerator(api_key="test-key")
    prior_history = [
        Message(role="user", content="이전 질문"),
        Message(role="assistant", content="이전 답변"),
    ]
    result = gen.generate(
        query="추가 질문",
        context_chunks=["관련 내용"],
        history=prior_history,
    )

    assert len(result.history) == 4
    assert result.history[0].content == "이전 질문"
    assert result.history[2].content == "추가 질문"


def test_generator_calls_chat_api(mock_openai):
    gen = OpenAIGenerator(api_key="test-key")
    gen.generate(query="질문", context_chunks=["컨텍스트"])

    gen._client.chat.completions.create.assert_called_once()
    call_kwargs = gen._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["temperature"] == 0.0


def test_generator_context_in_message(mock_openai):
    gen = OpenAIGenerator(api_key="test-key")
    gen.generate(query="질문", context_chunks=["청크1", "청크2"])

    messages = gen._client.chat.completions.create.call_args.kwargs["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "청크1" in user_msg["content"]
    assert "청크2" in user_msg["content"]
