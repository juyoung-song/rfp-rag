# 기술 기획서: RFP-RAG

## 1. 기술 스택 요약

| 영역 | 기술 |
|---|---|
| **LLM** | OpenAI gpt-4o-mini |
| **임베딩** | OpenAI text-embedding-3-small / Cohere embed-multilingual-v3.0 (비교 실험) |
| **Re-ranker** | Cohere Rerank (Phase 2~) |
| **Vector DB** | Chroma (Phase 1~3) → Supabase pgvector (Phase 4) |
| **백엔드** | Python / FastAPI |
| **프론트엔드** | React (Vite) |
| **컨테이너** | Docker / Docker Compose |
| **클라우드** | AWS (ECR + EC2/ECS + S3 + CloudFront) |

---

## 2. 프로젝트 디렉토리 구조

```
rfp-rag/
├── backend/
│   ├── pipeline/
│   │   ├── loader/             # PDF 파싱 모듈
│   │   │   ├── base.py         # BaseLoader 인터페이스
│   │   │   ├── pymupdf_loader.py
│   │   │   ├── pdfplumber_loader.py
│   │   │   └── opendataloader_loader.py
│   │   ├── chunker/            # 청킹 전략 모듈
│   │   │   ├── base.py
│   │   │   ├── fixed_chunker.py
│   │   │   ├── recursive_chunker.py
│   │   │   └── semantic_chunker.py
│   │   ├── embedder/           # 임베딩 모듈
│   │   │   ├── base.py
│   │   │   ├── openai_embedder.py
│   │   │   └── cohere_embedder.py
│   │   ├── retriever/          # 검색 모듈 (Phase별 교체)
│   │   │   ├── base.py
│   │   │   ├── naive_retriever.py      # Phase 1
│   │   │   ├── hybrid_retriever.py     # Phase 2
│   │   │   ├── corrective_retriever.py # Phase 3
│   │   │   └── agentic_retriever.py    # Phase 4
│   │   └── generator/          # 생성 모듈
│   │       ├── prompt_templates.py
│   │       └── generator.py
│   ├── db/
│   │   └── chroma_client.py
│   ├── api/
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   ├── ingest.py
│   │   │   └── documents.py
│   │   └── main.py
│   ├── experiments/            # 비교 실험 스크립트
│   │   ├── eval_parser.py
│   │   ├── eval_chunker.py
│   │   ├── eval_embedder.py
│   │   └── eval_retriever.py
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWindow.jsx
│   │   │   ├── MessageBubble.jsx
│   │   │   └── DocumentFilter.jsx
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── Dockerfile
│   └── package.json
│
├── data/                       # gitignore (원본 RFP 문서)
│   ├── *.pdf
│   └── data_list.csv
│
├── docs/
│   ├── project_plan.md         # 프로젝트 기획서
│   ├── tech_spec.md            # 기술 기획서 (현재 문서)
│   └── decisions/              # ADR 문서 (gitignore)
│
├── docker-compose.yml
├── .gitignore
└── CLAUDE.md
```

---

## 3. RAG 파이프라인 상세

### 3.1 인덱싱 파이프라인

```
PDF 파일
    ↓ [Loader] PDF 파싱 → 원문 텍스트 추출
    ↓ [Chunker] 청킹 → 텍스트 분할 (청크 크기, 오버랩 설정)
    ↓ [Embedder] 임베딩 생성 → 벡터화
    ↓ [Chroma] 벡터 + 메타데이터 저장
         metadata: {공고번호, 사업명, 발주기관, 사업금액, 마감일, 파일명, chunk_index}
```

### 3.2 쿼리 파이프라인 (Phase별)

**Phase 1 — Naive RAG**
```
사용자 질문
    → 질문 임베딩
    → Chroma 코사인 유사도 Top-K 검색
    → gpt-4o-mini 컨텍스트 기반 답변 생성
```

**Phase 2 — Hybrid RAG**
```
사용자 질문
    → 메타데이터 필터 파싱 (발주기관명, 사업명 감지)
    → BM25 키워드 검색 (병렬)
    → Dense 벡터 검색 (병렬)
    → RRF(Reciprocal Rank Fusion) 결합
    → Cohere Rerank 재정렬
    → gpt-4o-mini 답변 생성
```

**Phase 3 — Corrective RAG**
```
사용자 질문
    → Hybrid 검색
    → LLM-as-judge: 검색 결과 관련성 평가 (1~5점)
    → 점수 낮음(≤2): 쿼리 재작성 → 재검색
    → 점수 높음(≥3): 바로 생성 단계로
    → gpt-4o-mini 답변 생성
```

**Phase 4 — Agentic RAG**
```
사용자 질문
    → LangGraph 에이전트
        ├── Planner: 질문 분해 (서브쿼리 생성)
        ├── Retriever: 서브쿼리별 병렬 검색
        ├── Reasoner: 검색 결과 종합 추론
        └── Critic: 답변 품질 자기 평가 → 필요시 재검색
    → 최종 답변 생성
```

---

## 4. API 명세

### POST /api/ingest
PDF 문서를 파싱하고 벡터 DB에 인덱싱

**Request:**
```json
{
  "file_paths": ["string"],   // 파일 경로 목록 (전체 인덱싱 시 생략)
  "force_reindex": false
}
```

**Response:**
```json
{
  "indexed_count": 100,
  "elapsed_seconds": 45.2
}
```

---

### POST /api/chat
질문에 대한 RAG 기반 답변 생성

**Request:**
```json
{
  "message": "국민연금공단 이러닝시스템 요구사항 정리해줘",
  "session_id": "uuid",
  "filters": {
    "organization": "국민연금공단",
    "budget_min": null,
    "budget_max": null
  }
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": [
    {
      "document": "국민연금공단_이러닝시스템.pdf",
      "chunk": "string",
      "score": 0.92
    }
  ],
  "session_id": "uuid"
}
```

---

### GET /api/documents
인덱싱된 문서 목록 조회

**Response:**
```json
{
  "documents": [
    {
      "id": "string",
      "title": "사업명",
      "organization": "발주기관",
      "budget": 130000000,
      "deadline": "2024-10-15"
    }
  ],
  "total": 100
}
```

---

### GET /api/chat/history/{session_id}
특정 세션의 대화 히스토리 조회

---

### POST /api/evaluate
고정 질문 세트로 성능 평가 실행

---

## 5. 비교 실험 설계

### 5.1 PDF 파서 실험 (Phase 1 시작 전)

| 실험 대상 | PyMuPDF | pdfplumber | OpenDataLoader |
|---|---|---|---|
| 텍스트 추출 정확도 | - | - | - |
| 표 추출 품질 | - | - | - |
| 처리 속도 (100개 기준) | - | - | - |
| 한국어 처리 | - | - | - |

**평가 방법:** 동일 5개 문서 파싱 후 수동 검토 + 처리 시간 측정

---

### 5.2 청킹 전략 실험

| 전략 | 청크 크기 | 오버랩 | Recall@5 |
|---|---|---|---|
| Fixed size | 500 tokens | 50 | - |
| Recursive | 500 tokens | 50 | - |
| Semantic | 자동 | - | - |

**평가 방법:** 고정 질문 10개에 대한 Recall@5 측정

---

### 5.3 임베딩 모델 실험 (Phase 2)

| 모델 | Recall@5 | MRR | 처리 속도 | 비용 |
|---|---|---|---|---|
| text-embedding-3-small | - | - | - | - |
| embed-multilingual-v3.0 | - | - | - | - |

---

### 5.4 Retriever 실험 (Phase 2)

| 방식 | Recall@5 | MRR | 응답 속도 |
|---|---|---|---|
| Dense only (Phase 1) | - | - | - |
| BM25 only | - | - | - |
| Hybrid (RRF) | - | - | - |
| Hybrid + Rerank | - | - | - |

---

## 6. 배포 아키텍처

```
[사용자]
    ↓ HTTPS
[CloudFront + S3]  ← React 빌드 파일 (정적 호스팅)
    ↓ /api/* 요청
[EC2 or ECS]
    ├── FastAPI 컨테이너 (port 8000)
    └── Chroma 컨테이너 (port 8001) — Phase 1~3
         ↓ (Phase 4)
    └── Supabase (외부 클라우드 DB)
```

### Docker Compose (로컬 개발)
```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: .env
    volumes: ["./data:/app/data"]
    depends_on: [chroma]

  chroma:
    image: chromadb/chroma
    ports: ["8001:8001"]
    volumes: ["chroma_data:/chroma/chroma"]

  frontend:
    build: ./frontend
    ports: ["3000:3000"]

volumes:
  chroma_data:
```

---

## 7. 환경 변수

```env
# OpenAI
OPENAI_API_KEY=

# Cohere
COHERE_API_KEY=

# Chroma
CHROMA_HOST=chroma
CHROMA_PORT=8001

# Phase 4
SUPABASE_URL=
SUPABASE_KEY=

# 설정
ACTIVE_PHASE=1          # 현재 활성 Phase (1~4)
EMBEDDING_MODEL=openai  # openai | cohere
TOP_K=5
```

---

## 8. 기술 의사결정 문서 목록

| 파일 | 내용 |
|---|---|
| `docs/decisions/ADR-001-llm-embedding-api.md` | LLM, 임베딩 모델, Re-ranker 선택 |
| `docs/decisions/ADR-002-vector-db.md` | Vector DB 선택 및 마이그레이션 전략 |
| `docs/decisions/ADR-003-rag-architecture-roadmap.md` | 4단계 RAG 로드맵 |
| `docs/decisions/ADR-004-deployment.md` | Docker + AWS 배포 전략 |
| `docs/decisions/ADR-005-pdf-parser.md` | PDF 파서 실험 결과 및 최종 선택 (실험 후 작성) |
| `docs/decisions/ADR-006-chunking.md` | 청킹 전략 실험 결과 (실험 후 작성) |
| `docs/decisions/ADR-007-embedding-comparison.md` | 임베딩 모델 비교 결과 (실험 후 작성) |
