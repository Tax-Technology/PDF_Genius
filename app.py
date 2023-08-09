import streamlit as st
import tempfile
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain

class CharacterTextSplitter:
    def __init__(self, separator="\n", chunk_size=800, chunk_overlap=200, length_function=len):
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def split_text(self, text_content):
        return [text_content[i:i+self.chunk_size] for i in range(0, len(text_content), self.chunk_size-self.chunk_overlap)]

def summarize(text_splitter, pages, page_number):
    view = pages[page_number - 1]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summaries = chain.run(Document(page_content=t) for t in text_splitter.split_text(view.page_content))
    return summaries

def answer_question(pages, question, cache={}):
    if question in cache:
        return cache[question]
    chain = load_qa_chain(llm, chain_type="seq2seq")
    answer = chain.run(pages, question)
    cache[question] = answer
    return answer

if __name__ == "__main__":
    st.title("Elaine’s PDF Assistant")

    # Allow user to input their OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API key:")

    # Check if the API key is provided and valid
    if openai_api_key:
        try:
            llm = OpenAIEmbeddings(openai_api_key=openai_api_key, temperature=0)
        except Exception as e:
            st.error("Error: Invalid OpenAI API key. Please provide a valid key.")
            st.stop()

        pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

        if pdf_file is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(pdf_file.read())
                    pdf_path = tmp_file.name
                    reader = PdfReader(pdf_path)
                    pages = reader.pages
            except Exception as e:
                st.error("Error: Invalid PDF file. Please provide a valid PDF.")
                st.stop()

            page_selection = st.radio("Page selection", ["Single page", "Page range", "Overall Summary", "Question"])

            if page_selection == "Single page":
                page_number = st.number_input("Enter page number", min_value=1, max_value=len(pages), value=1, step=1)
                text_splitter = CharacterTextSplitter()
                summaries = summarize(text_splitter, pages, page_number)
                st.subheader("Summary")
                st.write(summaries)

            elif page_selection == "Page range":
                start_page = st.number_input("Enter start page", min_value=1, max_value=len(pages), value=1, step=1)
                end_page = st.number_input("Enter end page", min_value=start_page, max_value=len(pages), value=start_page, step=1)

                text_splitter = CharacterTextSplitter()
                summaries = summarize(text_splitter, pages, start_page-1)
                st.subheader("Summary")
                st.write(summaries)

            elif page_selection == "Overall Summary":
                text_splitter = CharacterTextSplitter()
                summaries = summarize(text_splitter, pages, 0)
                st.subheader("Summary")
                st.write(summaries)

            elif page_selection == "Question":
                question = st.text_input("Enter your question")
                answer = answer_question(pages, question)
                st.write("Answer:", answer)

    else:
        st.warning("Please provide your OpenAI API key to use this app.")
