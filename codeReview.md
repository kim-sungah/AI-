건설공사 사고 예방 및 대응책 생성 : 한솔데코 시즌 3 AI 경진대회👷
===============================================================
---------------------------------------------------------------

### 개발횐경
'''
Colab pro A100 GPU
python
'''

<hr/>

### 라이브러리 불러오기
'''
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
'''

### PDF 문서 로드
- PyPDFium2Loader : langchain_community.document_loaders에서 제공하는 PDF 문서 로더. PyPDFium2 라이브러리를 기반으로 PDF 문서를 읽고 텍스트 데이터를 추출하는 기능 제공.
- pdf_path : PDF 파일의 경로 저장
- loader : PyPDFium2Loader 객체를 생성하고, pdf_path를 전달
- load : load() 메서드를 실행하여 PDF 문서의 텍스트 로드. 반환값 load는 일반적으로 Document 객체들의 리스트이며, 각 객체는 PDF 페이지 또는 섹션을 나타냄.
'''
from langchain_community.document_loaders import PyPDFium2Loader

pdf_path = '/content/drive/MyDrive/Colab Notebooks/data./건설안전지침/건설공사 안전보건 설계 지침.pdf'

loader = PyPDFium2Loader(pdf_path)
load = loader.load()
'''

