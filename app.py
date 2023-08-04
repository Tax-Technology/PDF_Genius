import os
import streamlit as st
import tempfile
from PyPDF2 import PdfFileReader, PdfFileWriter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain

# Get your OpenAI API key from GitHub Codespaces secrets
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Apply pink unicorn theme
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(45deg, #f9a7b0, #fff6d6);
    }
    </style>
    """,
    unsafe_allow_html=True
)

class CharacterTextSplitter:
    def __init__(self, separator="\n", chunk_size=800, chunk_overlap=200, length_function=len):
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def split_text(self, text_content):
        chunks = []
        current_chunk = []
        for i in range(len(text_content)):
            if i % self.chunk_size == 0:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
            current_chunk.append(text_content[i])
            if i - self.chunk_overlap >= 0 and i % self.chunk_size == self.chunk_overlap - 1:
                current_chunk = []
        if current_chunk:
            chunks.append("".join(current_chunk))
        return chunks

llm = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, temperature=0)

def summarize(pages, page_number):
    view = pages[page_number - 1]
    texts = text_splitter.split_text(view.page_content)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summaries = chain.run(docs)
    return summaries

def answer_question(pages, question):
    chain = load_qa_chain(llm, chain_type="seq2seq")
    answer = chain.run(pages, question)
    return answer

if __name__ == "__main__":
    st.title("Elaine’s PDF Assistant")
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

    if pdf_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf_file.read())
            pdf_path = tmp_file.name
            reader = PdfFileReader(pdf_path)
            pages = reader.pages

        page_selection = st.radio("Page selection", ["Single page", "Page range", "Overall Summary", "Question"])

        if page_selection == "Single page":
            page_number = st.number_input("Enter page number", min_value=1, max_value=len(pages), value=1, step=1)
            summaries = summarize(pages, page_number)
            st.subheader("Summary")
            st.write(summaries)

        elif page_selection == "Page range":
            start_page = st.number_input("Enter start page", min_value=1, max_value=len(pages), value=1, step=1)
            end_page = st.number_input("Enter end page", min_value=start_page, max_value=len(pages), value=start_page, step=1)

            texts = []
            for page_number in range(start_page, end_page+1):
                view = pages[page_number-1]
                texts.append(view.page_content)
            summaries = summarize(texts, 0)
            st.subheader("Summary")
            st.write(summaries)

        elif page_selection == "Overall Summary":
            summaries = summarize(pages, 0)
            st.subheader("Summary")
            st.write(summaries)

        elif page_selection == "Question":
            question = st.text_input("Enter your question")
            answer = answer_question(pages, question)
            st.write("Answer:", answer)
