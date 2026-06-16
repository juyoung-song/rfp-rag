"""평가 유틸리티 함수 단위 테스트 (pytest)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from baseline_eval import keyword_hit_rate, summarize


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


def test_summarize_basic():
    results = [
        {"keyword_hit_rate": 1.0, "top_retrieved_score": 0.9, "latency_sec": 1.0},
        {"keyword_hit_rate": 0.5, "top_retrieved_score": 0.8, "latency_sec": 2.0},
    ]
    s = summarize(results)
    assert s["total_questions"] == 2
    assert abs(s["avg_keyword_hit_rate"] - 0.75) < 1e-6
    assert abs(s["avg_top_retrieved_score"] - 0.85) < 1e-6
    assert s["perfect_hit_count"] == 1
