# ===========================
# 📦 Imports
# ===========================
import streamlit as st
# from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import tempfile
import fitz  # PyMuPDF
import docx
import os
from typing import List, Tuple
import difflib
import torch

from dotenv import load_dotenv

load_dotenv()

os.environ["CURL_CA_BUNDLE"]=""
openai_api_key = os.getenv("OPENAI_API_KEY")

def load_file_content(file) -> str:
    ext = os.path.splitext(file.name)[1].lower()
    if ext == '.pdf':
        doc = fitz.open(stream=file.read(), filetype='pdf')
        return "\n\n".join(page.get_text() for page in doc)
    elif ext == '.docx':
        doc = docx.Document(file)
        return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    elif ext == '.txt':
        return file.read().decode('utf-8')
    else:
        st.error("Unsupported file type")
        return ""

def diff_srs(original: str, edited: str) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    original_lines = set(line.strip() for line in original.splitlines() if line.strip())
    edited_lines = set(line.strip() for line in edited.splitlines() if line.strip())
    added = list(edited_lines - original_lines)
    removed = list(original_lines - edited_lines)
    modified = []
    # Simple heuristic: modifications are lines that are removed but similar to an added one
    for rem in removed:
        for add in added:
            if rem.split()[:3] == add.split()[:3]:  # crude similarity
                modified.append((rem, add))
    for mod in modified:
        if mod[0] in removed: removed.remove(mod[0])
        if mod[1] in added: added.remove(mod[1])
    return added, removed, modified



def create_faiss_index(text: str):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": torch.device("cpu")})
    # embedding_model = OpenAIEmbeddings()
    print("embedding model loaded")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=['\n\n', '\n', '.', ',']
    )
    docs = splitter.create_documents([text])
    return FAISS.from_documents(docs, embedding_model), docs

def get_top_k_chunks(index: FAISS, query: str, k: int = 3):
    return index.similarity_search(query, k)


def analyze_change(change_text: str, sas_index: FAISS) -> str:
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o")
    prompt_template = (
    "You are a software architect assistant. Analyze the following requirement change and suggest updates "
    "needed in the given software architectural document section.\n\n"
    "Requirement Change:\n{change_text}\n\n"
    "Architecture Chunk:\n{sas_chunk_text}\n\n"
    "Your response should include:\n"
    "- Affected architecture section\n- Suggested changes\n- Rationale"
)
    top_chunks = get_top_k_chunks(sas_index, change_text)
    recommendations = []
    for chunk in top_chunks:
        prompt = prompt_template.format(change_text=change_text, sas_chunk_text=chunk.page_content)
        # qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=sas_index.as_retriever())
        print("Prompt:", prompt)
        response = llm.invoke(prompt)
        print(response.content)
        recommendations.append(response.content)
        # recommendations.append(response["answer"])
    return "\n\n".join(recommendations)

# ===========================
# 🚀 Streamlit Events
# ===========================
def main():
    st.set_page_config(layout="wide")
    st.title("🧠 GenAI Assistant for Software Architects")

    col1, col2 = st.columns(2)

    if 'original_srs' not in st.session_state:
        st.session_state.original_srs = ""
        st.session_state.srs_index = None
        st.session_state.srs_chunks = []

    if 'sas_text' not in st.session_state:
        st.session_state.sas_text = ""
        st.session_state.sas_index = None
        st.session_state.sas_chunks = []

    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []


    with col1:
        st.header("📘 Software Requirements")
        srs_file = st.file_uploader("Upload SRS Document", type=["pdf", "docx", "txt"], key="srs")
        if srs_file:
            with st.spinner("Parsing SRS file..."):
                print("Parsing SRS file...")
                srs_text = load_file_content(srs_file)
                st.session_state.original_srs = srs_text
            with st.spinner("creating faiss index..."):
                st.session_state.srs_index, st.session_state.srs_chunks = create_faiss_index(srs_text)
        srs_edited = st.text_area("Edit Requirements", st.session_state.original_srs, height=400, key="edited_srs")

    with col2:
        st.header("📙 Software Architecture Specs")
        sas_file = st.file_uploader("Upload SAS Document", type=["pdf", "docx", "txt"], key="sas")
        if sas_file:
            with st.spinner("Parsing SAS file..."):
                st.session_state.sas_text = load_file_content(sas_file)
            with st.spinner("creating faiss index..."):
                st.session_state.sas_index, st.session_state.sas_chunks = create_faiss_index(st.session_state.sas_text)
        st.text_area("Architecture Document", st.session_state.sas_text, height=400, disabled=True)

    
    if st.button("Analyze Changes"):
        print("Analyzed button clicked")
        if not st.session_state.original_srs or not st.session_state.sas_text:
            st.error("Please upload both SRS and SAS documents.")
            return
        
        added, removed, modified = diff_srs(st.session_state.original_srs, srs_edited)
        print("Diff is evaluated")
        if not added and not removed and not modified:
            st.warning("No changes detected.")
            return
        
        st.session_state.analysis_results = []
        
        if added:
            st.write("### Added Requirements")
            for item in added:
                st.write(f"- {item}")
                analysis = analyze_change(item, st.session_state.sas_index)
                st.session_state.analysis_results.append(analysis)
        
        if removed:
            st.write("### Removed Requirements")
            for item in removed:
                st.write(f"- {item}")
                analysis = analyze_change(item, st.session_state.sas_index)
                st.session_state.analysis_results.append(analysis)
        
        if modified:
            st.write("### Modified Requirements")
            for original, edited in modified:
                st.write(f"- Original: {original}\n- Edited: {edited}")
                analysis = analyze_change(edited, st.session_state.sas_index)
                st.session_state.analysis_results.append(analysis)
        
        if st.session_state.analysis_results:
            st.write("### Analysis Results")
            for result in st.session_state.analysis_results:
                st.write(result)


if __name__ == "__main__":
    main()
