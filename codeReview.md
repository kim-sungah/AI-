건설공사 사고 예방 및 대응책 생성 : 한솔데코 시즌 3 AI 경진대회👷
===============================================================
---------------------------------------------------------------

### 개발환경
```
Colab pro A100 GPU
python
```

<hr/>

### 라이브러리 불러오기
```
import pandas as pd
import tensorflow as tf
import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
```

### PDF 문서 로드
- PyPDFium2Loader : langchain_community.document_loaders에서 제공하는 PDF 문서 로더. PyPDFium2 라이브러리를 기반으로 PDF 문서를 읽고 텍스트 데이터를 추출하는 기능 제공.
- pdf_path : PDF 파일의 경로 저장
- loader : PyPDFium2Loader 객체를 생성하고, pdf_path를 전달
- load : load() 메서드를 실행하여 PDF 문서의 텍스트 로드. 반환값 load는 일반적으로 Document 객체들의 리스트이며, 각 객체는 PDF 페이지 또는 섹션을 나타냄.
```
from langchain_community.document_loaders import PyPDFium2Loader

pdf_path = '/content/drive/MyDrive/Colab Notebooks/data./건설안전지침/건설공사 안전보건 설계 지침.pdf'

loader = PyPDFium2Loader(pdf_path)
load = loader.load()
```

### 데이터 로드 및 전처리
- 데이터 전처리
1. 공사 종류 컬럼 분할 : 공사종류 열의 값을 '/' 기준으로 나누기 (대분류 or 중분류)
2. 공종 컬럼 분할 : 공종 열을 '>' 기준으로 나누기 (대분류 or 중분류)
3. 사고객체 컬럼 분할 : 사고객체 열을 '>' 기준으로 나누기 (대분류 or 중분류)
```
train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data./train.csv', encoding = 'utf-8-sig')
test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data./test.csv', encoding = 'utf-8-sig')

train['공사종류(대분류)'] = train['공사종류'].str.split('/').str[0]
train['공사종류(중분류)'] = train['공사종류'].str.split('/').str[1]
train['공종(대분류)'] = train['공종'].str.split('>').str[0]
train['공종(중분류)'] = train['공종'].str.split('>').str[1]
train['사고객체(대분류)'] = train['사고객체'].str.split('>').str[0]
train['사고객체(중분류)'] = train['사고객체'].str.split('>').str[1]

test['공사종류(대분류)'] = test['공사종류'].str.split('/').str[0]
test['공사종류(중분류)'] = test['공사종류'].str.split('/').str[1]
test['공종(대분류)'] = test['공종'].str.split('>').str[0]
test['공종(중분류)'] = test['공종'].str.split('>').str[1]
test['사고객체(대분류)'] = test['사고객체'].str.split('>').str[0]
test['사고객체(중분류)'] = test['사고객체'].str.split('>').str[1]
```

### 데이터프레임 기반 QA 형식 데이터셋 생성
1. 훈련 데이터
- 각 행에 대해 특정 필드를 조합하여 question과 answer 구조 만들기
- question : 사고와 관련된 정보를 자연어 문장으로 정리
- answer : 해당 사고의 '재발방지대책 및 향후조치계획' 저장
- combined_training_data : 질문과 답변이 포함된 Pandas DataFrame 형태로 변환
```
combined_training_data = train.apply(
     lambda row: {
        "question": (
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
            f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
        ),
        "answer": row["재발방지대책 및 향후조치계획"]
    },
    axis=1
)

# DataFrame으로 변환
combined_training_data = pd.DataFrame(list(combined_training_data))
```
   
2. 테스트 데이터
- 각 행에 대해 특정 필드를 조합하여 question 구조 만들기
- question : 사고와 관련된 정보를 자연어 문장으로 정리
- combined_test_data : 질문이 포함된 Pandas DataFrame 형태로 변환
```
combined_test_data = test.apply(
    lambda row: {
        "question": (
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
            f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
        )
    },
    axis=1
)

# DataFrame으로 변환
combined_test_data = pd.DataFrame(list(combined_test_data))
```

### 모델 설정
- BitsAndBytesConfig : Hugging Face transformers 라이브러리에서 제공하는 저비트 양자화 설정클래스(메모리 사용량 감소, 속도 향상, 성능 유지를 목적으로 사용)
- load_in_4bit = True : 모델을 4bit 양자화하여 로드(대형 모델을 로컬 환경에서 실행하기 위함).
- bnb_4bit_use_double_quant = True : 이중 양자화(효율적인 압축 가능).
- bnb_4bit_quant_type = "nf4" : Normal Float 4(LLM 성능 저하를 최소화하면서 메모리 절약 가능).
- bnb_4bit_compute_dtype = torch.bfloat16 : NVIDIA A100, H100 등 최신 GPU에 최적화
```
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)
```

### LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct 모델
- LG AI Research에서 개발한 대형 언어 모델
- 모델 사이즈 : 32B, 7.8B, 2.4B
- 장점 : 저사양 GPU에서도 훈련 및 배포 가능, long-context processing 가능, 최대 32K 토큰의 긴 문맥 처리 가능
- 한국어 및 영어 지원
```
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model_id = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
```

### FAISS 기반 벡터 스토어 생성
1. 훈련 데이터 준비
- conbined_training_data에서 질문과 답변 리스트 가져옴.
- tolist()를 사용해 Pandas 시리즈를 Python 리스트로 변환.
- 각 질문과 답변을 Q&A 포맷의 텍스트 문서로 변환.
- zip(train_questions_prevention, train_answers_prevention) : 질문과 답변을 쌍으로 처리
2. 임베딩 모델 정의
- jhgan/ko-sbert-nli : Sentence-BERT 기반의 한국어 문장 임베딩 모델(=Hugging Face에서 제공하는 한글 문장 의미 분석 최적화 모델).
3. 벡터 스토어 생성
- FAISS.from_texts() : 텍스트 데이터를 벡터화하여 FAISS 저장소에 추가
- FAISS : 대규모 벡터 검색을 효율적으로 수행하는 라이브러리. 유사한 문서를 빠르게 찾기 위한 벡터 인덱싱 지원.
- 훈련 데이터를 벡터로 변환하여 FAISS에 저장
4. Retriever 정의(유사 문서 검색)
```
# vector store 생성
# Train 데이터 준비
train_questions_prevention = combined_training_data['question'].tolist()
train_answers_prevention = combined_training_data['answer'].tolist()

train_documents = [
    f"Q: {q1}\nA: {a1}"
    for q1, a1 in zip(train_questions_prevention, train_answers_prevention)
]

# 임베딩 생성
embedding_model_name = "jhgan/ko-sbert-nli"  # 임베딩 모델 선택
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

# 벡터 스토어에 문서 추가
vector_store = FAISS.from_texts(train_documents, embedding)

# Retriever 정의
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
```

### RAG 체인 생성
1. 텍스트 생성 파이프라인 설정
- pipeline() : Hugging Face transformers 라이브러리에서 제공하는 텍스트 생성 파이프라인
- text-generation : 텍스트 생성
- do_sample =  True : 생성된 텍스트에서 확률적으로 단어 선택 (랜덤성)
- temperature = 0.1 : 낮은 값일수록 더 결정론적인(정확한) 답변 생성
- return_full_text = False : 입력 프롬프트는 출력에 포함하지 않음
- max_new_tokens = 64 : 최대 64개 토큰까지 출력
```
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,  # sampling 활성화
    temperature=0.1,
    return_full_text=False,
    max_new_tokens=64,
)
```
2. 프롬프트 템플릿 정의
```
prompt_template = """
### 지침: 당신은 건설공사 사고 상황 데이터를 바탕으로 사고 원인을 분석하고 재발방지 대책을 포함한 대응책을 자동으로 생성하는 AI 모델입니다.
질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.
- 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
- '다음과 같은 조치를 취할 것을 제안합니다:' 와 같은 내용을 포함하지 마세요.
- 입력받은 내용에 대해 추가, 삭제, 변경 없이 받은 내용만을 그대로 출력합니다.

{context}

### 질문:
{question}

[/INST]

"""
```
3. LLM 연결 + 프롬프트 객체 생성
- HuggingFacePipeline()을 사용해 텍스트 생성 파이프라인을 LLM 모델로 변환
- 이후 체인에서 LLM 호출할 수 있도록 래핑
- 프롬프트 템플릿을 LangChain에서 사용할 수 있도록 객체화
```
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# 커스텀 프롬프트 생성
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)
```
4. RAG 체인 생성
- RetrievalQA.from_chain_type()을 사용해 검색 기반 QA 체인 생성
- chain_type = 'stuff' : 검색된 문서를 단순 결합하여 답변 생성
- return_source_documents = True : 참고 문서(출처) 반환
- chain_type_kwargs = {"prompt" : prompt} : 사용자 정의 프롬프트 적용
```
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 단순 컨텍스트 결합 방식 사용
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}  # 커스텀 프롬프트 적용
)
```

### AI 모델 답변 생성 및 저장
- batch : 병렬 실행. GPU 사용 시 병렬 연산 초적화 가능
- batch_size : 배치 단위로 모델을 실행하여 속도 + 메모리 효율성 증가
- qa_chain.batch(batch_questions) : 배치 크기만큼 한 번에 여러 개의 질문 처리
```
test_results = []

print("테스트 실행 시작... 총 테스트 샘플 수:", len(combined_test_data))

# 배치 크기 설정 (예: 16)
batch_size = 16

# 데이터 배치 단위로 나누어 처리
for start_idx in range(0, len(combined_test_data), batch_size):
    end_idx = min(start_idx + batch_size, len(combined_test_data))
    batch_questions = combined_test_data['question'][start_idx:end_idx].tolist()

    print(f"\n[샘플 {start_idx + 1}~{end_idx}/{len(combined_test_data)}] 진행 중...")

    # 배치 단위로 모델 호출
    batch_results = qa_chain.batch(batch_questions)  # 🔥 병렬 처리 가능하도록 batch 사용

    # 결과 저장
    batch_texts = [res['result'] for res in batch_results]
    test_results.extend(batch_texts)

print("\n테스트 실행 완료! 총 결과 수:", len(test_results))
```

### AI 모델 평가 (벡터화)
- SentenceTransformer(embedding_model_name) : Hugging Face에서 SBERT 모델 로드
- jhgan/ko-sbert-sts : 한국어 텍스트 유사도 분석에 최적화된 SBERT 모델.
```
# submission
from sentence_transformers import SentenceTransformer

embedding_model_name = "jhgan/ko-sbert-sts"
embedding = SentenceTransformer(embedding_model_name)

# 문장 리스트를 입력하여 임베딩 생성
pred_embeddings = embedding.encode(test_results)
print(pred_embeddings.shape)  # (샘플 개수, 768)
```
