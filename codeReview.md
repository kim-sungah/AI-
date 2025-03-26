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

