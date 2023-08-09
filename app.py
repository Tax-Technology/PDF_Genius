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

class MyLLMChain(LLMChain):
    def __init__(self, llm):
        super().__init__(llm)

    def agenerate_prompt(self, prompt):
        return self.llm.generate_prompt(prompt)

    def apredict(self, documents):
        return self.llm.predict(documents)

    def apredict_messages(self, documents):
        return self.llm.predict_messages(documents)

    def generate_prompt(self, prompt):
        return self.llm.generate_prompt(prompt)

    def invoke(self, documents, prompt):
        return self.llm.invoke(documents, prompt)

    def predict(self, documents):
        return self.llm.predict(documents)

    def predict_messages(self, documents):
        return self.llm.predict_messages(documents)

if __name__ == "__main__":
    st.title("Elaine’s PDF Assistant")

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

        openai_api_key = st.text_input("Enter your OpenAI API key:")
        if openai_api_key:
            try:
                llm = OpenAIEmbeddings(openai_api_key=openai_api_key, temperature=0)
            except Exception as e:
                st.error("Error: Invalid OpenAI API key. Please provide a valid key.")
                st.stop()

            text_splitter = CharacterTextSplitter()

            st.subheader("Prompt:")
            prompt = st.text_input("Enter your prompt:")

            if st.button("Summarize"):
                if prompt:
                    summaries = summarize(llm, text_splitter, pages, 0)
                    st.subheader("Summaries")
                    st.write(summaries)
                else:
                    st.warning("Please provide a prompt.")

            if st.button("Ask Question"):
                if prompt:
                    chain = MyLLMChain(llm)
                    answer = answer_question(chain, pages, prompt)
                    st.subheader("Answer")
                    st.write(answer)
                else:
                    st.warning("Please provide a prompt.")

