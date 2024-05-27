# 개발내용

- LLM과 DPR(vector retrieval)을 refactoring 및 합치는 과정을 통해 QA task RAG pipeline제작

# Reference

- [LLM](https://github.com/nlpai-lab/KULLM)
- [DPR](https://github.com/TmaxEdu/KorDPR)


# 기능

- LLM을 fine-tuning 할 수 있다. huggingface에 있는 LLM이면 학습 가능하다.(default: KULLM_v2_12.8b)
- DPR을 fine-tuning 할 수 있다. encoder 2개를 사용했으며 2개다 klue/bert-base를 tokenizer로 사용한다. bert계열이면 교체 가능핟. (default: klue/bert-base)
- user input(질문)입력시 user input과 유사한 참조문서를 검색(cosine similarity)및 참조하여 답변한 정보를 response로 return하는 서버(flask)를 띄울 수 있다.
- user input(질문)을 입력하고 response를 화면에 출력하는 데모 페이지(streamlit)를 띄울 수 있다.
- fine-tining 결과를 docker image로 build 할 수 있다.
- docker-compose.yml을 사용해 container작동 후 자동으로 서버와 데모 페이지를 띄울 수 있다.

# 파일 구조

├─해당언어
│  │  README.md
│  ├─docs
│  │      01-문서.md
│  │      02-문서.md
│  └─images
│          이미지
│          이미지
