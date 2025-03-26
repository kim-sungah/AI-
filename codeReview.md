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
- 각 행에 대해 특정 필드를 조합하여 question과 answer 구조 만들기
- question : 사고와 관련된 정보를 자연어 문장으로 정리
- answer : 해당 사고의 '재발방지대책 및 향후조치계획' 저장
- combined_data : 질문과 답변이 포함된 Pandas DataFrame 형태로 변환
1. 훈련 데이터
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
