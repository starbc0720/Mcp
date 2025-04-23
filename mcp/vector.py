from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import os
import pandas as pd

# 현재 스크립트가 위치한 디렉토리로 작업 디렉토리 변경
cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)

# Chroma DB 저장 경로 설정 (data/chroma_db)
db_dir = os.path.join(cur_dir, "data", "chroma_db")
os.makedirs(db_dir, exist_ok=True)  # 경로 없으면 생성

# CSV 로딩
df = pd.read_csv("realistic_restaurant_reviews.csv")

# 임베딩 모델 초기화
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# documents 추가 여부 판단
add_documents = not os.listdir(db_dir)  # 디렉토리가 비어 있으면 추가

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

# 벡터 스토어 초기화
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_dir,
    embedding_function=embeddings
)

# 문서 추가
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# 검색기 생성
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}  # 오타 수정: search_kwars → search_kwargs
)