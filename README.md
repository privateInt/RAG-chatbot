# 개발내용

- LLM과 DPR(vector retrieval)을 refactoring 및 합치는 과정을 통해 QA task RAG pipeline제작
- 이 project에서는 관세에 관련된 주제를 다루며 데이터 형식을 맞춘다면 다른 domain에도 적용 가능

# Reference

- [LLM](https://github.com/nlpai-lab/KULLM)
- [DPR](https://github.com/TmaxEdu/KorDPR)


# 기능

- LLM을 fine-tuning 할 수 있다. huggingface에 있는 LLM이면 학습 가능하다.(default: KULLM_v2_12.8b)
- DPR을 fine-tuning 할 수 있다. encoder 2개를 사용했으며 모두 klue/bert-base를 tokenizer로 사용한다. bert계열이면 교체 가능하다. (default: klue/bert-base)
- user input(질문)입력시 user input과 유사한 참조문서를 검색(cosine similarity)및 참조하여 답변한 정보를 response로 return하는 서버(flask)를 띄울 수 있다.
- user input(질문)을 입력하고 response를 화면에 출력하는 데모 페이지(streamlit)를 띄울 수 있다.
- fine-tining 결과 및 서버, 데모 페이지를 docker image로 build 할 수 있다.
- docker-compose.yml을 사용해 container작동 후 자동으로 서버와 데모 페이지를 띄울 수 있다.

# 파일 구조
```sh
project
├── app.py
├── data
│   ├── kullm_custom_data_240228_for_retriever_dst.p
│   ├── kullm_custom_data_240228_for_retriever_src.p
│   ├── kullm_custom_data_240228.json
│   ├── kullm_custom_data_240228_passage_index.dpr
│   ├── kullm_custom_data_240228_passage_index_meta.dpr
│   ├── kullm_custom_data_240228_test_0.2.p
│   ├── kullm_custom_data_240228_train_0.8.p
│   └── kullm_custom_data_240228.xlsx
├── docker-compose.yml
├── Dockerfile
├── DPR
│   ├── DPR_data.py
│   ├── DPR_inference.py
│   ├── DPR_make_passage_vector.py
│   ├── DPR_model.py
│   ├── DPR_test.py
│   ├── DPR_trainer.py
├── LLM
│   ├── LLM_extract_data.py
│   ├── LLM_prompter.py
│   ├── LLM_trainer.py
│   └── templates
│       └── kullm.json
├── requirements.txt
├── run.sh
├── server.py
└── utils.py
```

# 폴더 및 파일 역할
| 폴더 및 파일 | 설명 |
|------|--------|
|data|fine-tuning 및 RAG 대상 파일을 저장하는 폴더, 현재 xlsx파일을 읽고 처리하는 방식이며, xlsx파일의 column에는 출처,제시문,질문,답변이 포함돼야 한다.
|DPR|DPR 관련 코드 저장|
|DPR/DPR_data.py|데이터 parsing 및 pytorch dataloader에 load할 수 있게 가공하여 pickle로 저장한다. (train.p, test.p, for_retriever_src.p)|
|DPR/DPR_model.py|DPR model을 정의한 파일, 모델은 질문(query)과 제시문(passage)를 각각의 encoder로 처리한다.|
|DPR/DPR_trainer.py|데이터를 사용해 DPR model을 fine-tuning한다. tensorboard 및 log 기록 기능을 추가해 loss를 확인할 수 있다. loss는 CE를 사용했다.|
|DPR/DPR_make_passage_vector|fine-tuned DPR model을 사용해 RAG target data를 vector로 변환해 faiss(vector DB)에 저장한다. (passage_index.dpr, passage_index_meta.dpr, for_retriever_dst.p)|
|DPR/DPR_test.py|fine-tuned DPR model의 정확도를 확인한다. 계산방법은 질문 입력시 매칭하는 제시문(passage)이 GT와 일치하는지 비교한다. top_k는 1,3의 경우에 대해 각각 진행하였다.|
|LLM|LLM 관련 코드 저장|
|LLM/LLM_extract_data.py|data폴더에 저장된 xlsx파일을 fine-tuning 가능하도록 json에 intruction, input, output형태로 저장한다.|
|LLM/LLM_prompter.py|LLM fine-tuning 및 inference시 입력 데이터를 지정한 prompt형태로 변경한다.|
|LLM/LLM_trainer.py|LLM을 fine-tuning하여 결과를 bin파일로 저장한다.|
|templates|LLM fine-tuning 및 inference시 사용하는 prompt를 저장하는 폴더|
|templates/kullm.json|prompt engineering에 대한 정보가 담겨 입력 데이터를 가공하는 json파일|
|requirements.txt|project 작동시 필요한 library를 모아놓은 txt파일|
|server.py|DPR, LLM의 fine-tuning 결과를 이용해 inference server(flask)를 작동|
|app.py|inference server에 입력값을 전송하고 return값을 출력하는 데모 페이지(streamlit)작동|
|run.sh|inference server와 데모 페이지를 한번에 작동하는 shell script 명령어 파일|
|Dokcerfile|project 결과를 docker image로 build할 수 있도록 명령어들을 모아놓은 파일|
|docker-compose.yml|docker image가 build되면 바로 사용할 수 있도록 mount, gpu, 명령어 등을 정의해놓은 yml파일|

# 하드웨어 스펙
- GPU: A100(40GiB)GPU * 4
- 저장 모델 용량: DPR 2.5GB, LLM 9.5GB

# 학습 실험

## DPR
- hyper parameter
  
| hyper parameter | value |
|------|--------|
|epochs|20|
|batch_size|16|
|learning_rate|3e-4|
|passage_chunk_size|100|

- GPU memory usage(fine-tuning): 약 25GiB
- GPU memory usage(inference): 약 4GiB
- DPR 논문에 의하면 batch_size를 크게할 수록 negative sample을 증가시켜 fine-tuning 성능을 향상시킬 수 있다.
- DPR 논문에 의하면 passage를 chunk_size로 분할하여 검색한다. 이 project에서는 정보 손실을 고려하여 분할된 chunk를 원복하여 LLM에 제공한다.

## LLM
- hyper parameter
  
| hyper parameter | value |
|------|--------|
|lora_r|32|
|lora_alpha|lora_r * 2|
|epochs|20|
|batch_size|128|
|micro_batch_size|8|
|learning_rate|3e-5|

- GPU memory usage(fine-tuning): 약 60GiB
- GPU memory usage(inference): 약 56GiB
- lora값에 따라 trainable parameter가 결정됐다. lora논문에 의하면 trainable parameter가 0.2%여도 성능에 큰 차이가 없다는 사실을 확인하기 위해, lora_r의 값은 32, 3072로 조정하여 각 trainable parameter를 0.2%, 16.3%로 변경하여 fine-tuning 성능을 비교했지만 큰 차이를 발견할 수 없었다.
- 데이터 수와 품질의 관계를 파악하기 위해 데이터 수가 많지만 품질이 떨어지는 데이터셋, 데이터 수가 적지만 품질이 좋은 데이터셋을 각각 fine-tuning하여 성능을 비교했다. 비교 결과 데이터 수가 적더라도 품질이 뛰어나야 LLM의 성능에 긍정적이라는 사실을 확인했다.

# 명령어

<table border="1">
  <tr>
    <th>내용</th>
    <th>명령어</th>
  </tr>
  <tr>
    <td rowspan="3">Row 1, Cell 2 (Rowspan 3)</td>
    <td>Row 1, Cell 3</td>
  </tr>
    <!-- Cell 2 is merged with the cell above -->
    <td>Row 2, Cell 3</td>
  </tr>
  <tr>
    <!-- Cell 2 is merged with the cell above -->
    <td>Row 3, Cell 3</td>
  </tr>
</table>
