import pytest
from pipeline.chunker.fixed_chunker import FixedChunker
from pipeline.chunker.recursive_chunker import RecursiveChunker
from pipeline.chunker.markdown_header_chunker import MarkdownHeaderChunker

SAMPLE_TEXT = """# 국민연금공단 이러닝시스템 운영 용역

## 1. 사업 개요

본 사업은 국민연금공단의 이러닝시스템을 운영하기 위한 용역으로, 학습 콘텐츠 제공 및 시스템 유지보수를 포함한다.

## 2. 주요 요구사항

| 항목 | 요구사항 |
|---|---|
| 가용성 | 99.9% 이상 |
| 응답시간 | 3초 이내 |
| 헬프데스크 | 평일 09:00-18:00 |

## 3. 기술 요구사항

가. 콘텐츠 관리: 기존 콘텐츠의 지속적인 업데이트 및 신규 콘텐츠 개발
나. 시스템 안정성: 99.9% 이상의 가동률 유지
다. 사용자 지원: 평일 09:00-18:00 헬프데스크 운영
""" * 8


def test_fixed_chunker_returns_list():
    chunker = FixedChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.split(SAMPLE_TEXT)
    assert isinstance(chunks, list)
    assert len(chunks) > 1


def test_fixed_chunker_respects_size():
    chunker = FixedChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.split(SAMPLE_TEXT)
    for chunk in chunks:
        assert len(chunk) <= 300


def test_recursive_chunker_returns_list():
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split(SAMPLE_TEXT)
    assert isinstance(chunks, list)
    assert len(chunks) > 1


def test_recursive_chunker_preserves_content():
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split(SAMPLE_TEXT)
    combined = " ".join(chunks)
    assert "국민연금공단" in combined
    assert "이러닝시스템" in combined


def test_markdown_header_chunker_returns_list():
    chunker = MarkdownHeaderChunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.split(SAMPLE_TEXT)
    assert isinstance(chunks, list)
    assert len(chunks) > 1


def test_markdown_header_chunker_splits_by_section():
    chunker = MarkdownHeaderChunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.split(SAMPLE_TEXT)
    combined = " ".join(chunks)
    # 주요 섹션 내용이 보존되는지 확인
    assert "사업 개요" in combined
    assert "주요 요구사항" in combined
