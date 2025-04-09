## ✅ LangChain이란?

> **LangChain은 LLM (Large Language Model)을 쉽게 활용하고 연결할 수 있도록 도와주는 Python 기반 오픈소스 프레임워크**입니다.

즉, 단순히 질문-답변만 하는 LLM(ChatGPT 등)을 넘어서서,
- **외부 데이터 검색**
- **파일 분석**
- **데이터베이스 질의**
- **문서 기반 QA**
- **툴과의 연동 (계산기, Google 검색, API 호출 등)**

…같은 **“실제 응용 프로그램”**을 만들 수 있도록 도와주는 “연결 도우미”라고 보면 됩니다.

---

## ✅ LangChain 이름 뜻?

- **Lang**: Language (LLM)
- **Chain**: 여러 작업들을 **체인처럼 연결**해서 처리

예: "사용자 질문 → 문서 검색 → 요약 → 답변 생성"  
이 전체 흐름을 하나의 체인으로 구성할 수 있어요.

---

## ✅ LangChain 주요 기능

| 기능 | 설명 |
|------|------|
| **Prompt Templates** | 프롬프트(질문)를 템플릿화해서 다양한 입력 처리 |
| **Chains** | 여러 단계의 LLM 작업을 연결해서 처리 |
| **Agents** | LLM이 도구(API, 계산기 등)를 선택해 자동으로 실행 |
| **Tools** | 외부 도구 연결 (예: Google 검색, SQL, Python 실행 등) |
| **Memory** | 대화의 문맥 기억 (기억력 탑재) |
| **VectorStores** | VectorDB와 연동 (Pinecone, Chroma 등) |
| **Retrieval** | 문서나 DB에서 정보 검색 후 LLM에게 전달 (→ RAG 구성 가능) |

---

## ✅ 어디에 쓰일까?

- 문서 기반 챗봇 (PDF, Notion, Google Docs 등)
- 개인 비서 앱 (질문하면 자동으로 찾아줌)
- LLM + 데이터베이스 연동 서비스
- ChatGPT 플러그인과 유사한 앱 만들기

---

## ✅ 예시: 간단한 문서 QA 시스템

```python
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 문서 불러오기 → 임베딩 → VectorDB 저장
loader = TextLoader("my_doc.txt")
docs = loader.load()
vectordb = Chroma.from_documents(docs, OpenAIEmbeddings())

# 질문 → 관련 문서 검색 → 답변 생성
qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=vectordb.as_retriever())
qa.run("이 문서에서 주요 내용이 뭐야?")
```

---

## ✅ LangChain vs 다른 프레임워크

| 프레임워크 | 특징 |
|------------|------|
| **LangChain** | 가장 유명, Python 중심, 기능 풍부 |
| **LlamaIndex** | 문서 중심의 RAG에 특화 |
| **Haystack** | RAG/QA 특화, FastAPI 기반 |
| **Semantic Kernel (MS)** | .NET 기반, 엔터프라이즈에 강점 
