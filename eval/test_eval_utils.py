"""평가 유틸리티 함수 단위 테스트 (pytest)"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from baseline_eval import keyword_hit_rate, summarize, llm_judge


# ── keyword_hit_rate ──────────────────────────────────────────────────────────

def test_keyword_hit_rate_all_present():
    rate = keyword_hit_rate("사업 예산은 5억원입니다.", ["예산", "억", "원"])
    assert rate == 1.0


def test_keyword_hit_rate_partial():
    rate = keyword_hit_rate("사업 예산은 5억원입니다.", ["예산", "없는단어"])
    assert rate == 0.5


def test_keyword_hit_rate_none():
    rate = keyword_hit_rate("관계없는 텍스트", ["예산", "억"])
    assert rate == 0.0


def test_keyword_hit_rate_empty_keywords():
    rate = keyword_hit_rate("텍스트", [])
    assert rate == 0.0


# ── summarize ─────────────────────────────────────────────────────────────────

def test_summarize_without_judge():
    results = [
        {"keyword_hit_rate": 1.0, "top_retrieved_score": 0.9, "latency_sec": 1.0},
        {"keyword_hit_rate": 0.5, "top_retrieved_score": 0.8, "latency_sec": 2.0},
    ]
    s = summarize(results)
    assert s["total_questions"] == 2
    assert abs(s["avg_keyword_hit_rate"] - 0.75) < 1e-6
    assert abs(s["avg_top_retrieved_score"] - 0.85) < 1e-6
    assert s["perfect_keyword_hit_count"] == 1
    assert "avg_judge_score" not in s


def test_summarize_with_judge():
    results = [
        {"keyword_hit_rate": 1.0, "top_retrieved_score": 0.9, "latency_sec": 1.0,
         "judge_score": 1.0, "judge_score_raw": 3, "judge_reason": "정확"},
        {"keyword_hit_rate": 0.5, "top_retrieved_score": 0.8, "latency_sec": 2.0,
         "judge_score": 0.333, "judge_score_raw": 1, "judge_reason": "부분 정확"},
    ]
    s = summarize(results)
    assert "avg_judge_score" in s
    assert s["perfect_judge_count"] == 1
    assert abs(s["avg_judge_score"] - (1.0 + 0.333) / 2) < 1e-3


# ── llm_judge ─────────────────────────────────────────────────────────────────

def test_llm_judge_returns_score_and_reason():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(
            content='{"score": 3, "reason": "정확한 답변입니다."}'
        ))]
    )
    result = llm_judge(mock_client, "질문", ["컨텍스트"], "답변")
    assert result["score"] == 3
    assert "정확한" in result["reason"]


def test_llm_judge_clamps_score():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(
            content='{"score": 5, "reason": "범위 초과"}'
        ))]
    )
    result = llm_judge(mock_client, "질문", ["컨텍스트"], "답변")
    assert result["score"] == 3  # 최대 3으로 클램핑


def test_llm_judge_zero_score():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(
            content='{"score": 0, "reason": "찾을 수 없음"}'
        ))]
    )
    result = llm_judge(mock_client, "질문", ["컨텍스트"], "찾을 수 없습니다.")
    assert result["score"] == 0
