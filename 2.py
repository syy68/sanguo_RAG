import streamlit as st
import os
import re
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. 页面配置 ---
st.set_page_config(page_title="知识问答· 三国志", page_icon="📜")

# --- 2. 路径配置（避开中文路径） ---
# 使用一个简单的、不含中文的路径存放索引
INDEX_SAVE_DIR = "D:/sanguo_db"
# 确保书稿路径正确（假设三国志.txt和代码在同一个文件夹）
BASE_DIR = Path(__file__).parent.absolute()
BOOK_PATH = str(BASE_DIR / "三国志.txt")


# --- 3. 核心引擎 (带强力目录创建) ---
@st.cache_resource
def init_rag_engine():
    # 你的 API 配置
    API_KEY = "ec405cb7-0eb5-4548-9877-9e5b96a2924b"
    BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    MODEL_ID = "ep-20260310140824-nv4lh"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(BASE_DIR / "model_cache")
    )

    # 检查索引文件是否存在
    faiss_file = os.path.join(INDEX_SAVE_DIR, "index.faiss")

    if os.path.exists(faiss_file):
        # 存在则加载
        vectorstore = FAISS.load_local(INDEX_SAVE_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        # 不存在则构建
        if not os.path.exists(BOOK_PATH):
            st.error(f"找不到文件：{BOOK_PATH}")
            st.stop()

        loader = TextLoader(BOOK_PATH, encoding="utf-8")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(splits, embeddings)

        # 【强力创建目录】
        if not os.path.exists(INDEX_SAVE_DIR):
            os.makedirs(INDEX_SAVE_DIR, exist_ok=True)

        # 保存索引
        vectorstore.save_local(INDEX_SAVE_DIR)

    llm = ChatOpenAI(openai_api_key=API_KEY, openai_api_base=BASE_URL, model=MODEL_ID, temperature=0.3)

    template = "你是一位历史学家。请根据史料回答：\n{context}\n问题：{question}"
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    return vectorstore, chain


# --- 4. 界面交互 ---
st.title("📜 知识问答· 三国智能助手")

with st.spinner("系统初始化中..."):
    vectorstore, chain = init_rag_engine()

query = st.text_input("请输入关于《三国志》的问题：")

if query:
    with st.spinner("查阅史料中..."):
        docs = vectorstore.similarity_search(query, k=4)
        context = "\n".join([d.page_content for d in docs])
        response = chain.invoke({"context": context, "question": query})

        st.markdown("### 🏛️ 史家评述")
        st.info(response.content)

        with st.expander("点击查看文献证据"):
            for doc in docs:
                st.write(doc.page_content)
                st.divider()