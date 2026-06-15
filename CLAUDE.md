# 프로젝트: rfp_rag

## 응답 규칙
- 항상 **한국어**로 답변
- 다음 단계 제안 또는 개발 시작 시 **항상 3가지 선택지** 제시

## 프로젝트 개요
- RFP(제안요청서) 문서를 RAG(Retrieval-Augmented Generation)로 처리하는 시스템
- 자세한 내용: `docs/project_plan.md`

## 기술 스택
- LLM: OpenAI gpt-4o-mini
- 임베딩: OpenAI text-embedding-3-small / Cohere embed-multilingual-v3.0
- Re-ranker: Cohere Rerank
- Vector DB: Chroma (Phase 1~3) → Supabase (Phase 4)
- 백엔드: FastAPI
- 프론트엔드: React (Vite)
- 배포: Docker + AWS
- 자세한 내용: `docs/tech_spec.md`

## 주요 문서
- 프로젝트 기획서: `docs/project_plan.md`
- 기술 기획서: `docs/tech_spec.md`
- 기술 의사결정 문서: `docs/decisions/` (로컬 전용, GitHub 비공개)

## Git 커밋 규칙
- 커밋 메시지는 **한국어**로 작성
- **Conventional Commits** 형식 사용
  - `feat:` 새로운 기능
  - `fix:` 버그 수정
  - `refactor:` 리팩토링
  - `docs:` 문서 수정
  - `chore:` 빌드/설정 등 기타
- 제목은 **50자 이내**
- 본문에 **변경 이유** 포함

예시:
```
feat: RFP 문서 업로드 API 추가

사용자가 PDF 형식의 RFP 문서를 업로드할 수 있도록
엔드포인트를 추가함. 향후 벡터 임베딩 처리와 연동 예정.
```

## 브랜치 전략
- `main` — 배포 가능한 안정 브랜치 (직접 push 금지)
- `feat/기능명` — 새 기능 개발
- `fix/버그명` — 버그 수정
- `docs/내용` — 문서 작업

## 이슈 전략
- **새 기능, 버그, 리팩토링 계획** 시 이슈를 먼저 생성
- 이슈 제목은 명사형으로 간결하게 (예: "RFP 파일 업로드 기능 구현")
- 라벨 사용: `feat`, `bug`, `refactor`, `docs`
- 아주 사소한 수정(오타, 주석 등)은 이슈 생략 가능

## PR 전략
- 1 PR = 1 이슈 = 1 기능
- PR 제목: `[feat] RFP 문서 업로드 기능 추가`
- PR 본문 포함 항목:
  - 무엇을, 왜 만들었는지
  - 테스트 방법
  - 관련 이슈 번호 (`Closes #이슈번호`)
- self-review 후 merge
