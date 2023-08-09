import streamlit as st
import openai
import tempfile
from PyPDF2 import PdfReader

# Set your OpenAI API key
openai_api_key = "YOUR_OPENAI_API_KEY"

# Initialize the OpenAI API client
openai.api_key = openai_api_key

def summarize(text, max_tokens=150):
    response = openai.Completion.create(
        engine="davinci",
        prompt=text,
        max_tokens=max_tokens
    )
    return response.choices[0].text.strip()

def answer_question(pages, question):
    context = " ".join([page.extract_text() for page in pages])
    response = openai.Answer.create(
        model="davinci",
        question=question,
        documents=[context]
    )
    return response.answers[0]

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

        page_selection = st.radio("Page selection", ["Single page", "Page range", "Overall Summary", "Question"])

        if page_selection == "Single page":
            page_number = st.number_input("Enter page number", min_value=1, max_value=len(pages), value=1, step=1)
            page_content = pages[page_number - 1].extract_text()
            summary = summarize(page_content)
            st.subheader("Summary")
            st.write(summary)
            st.button("Clear")

        elif page_selection == "Page range":
            start_page = st.number_input("Enter start page", min_value=1, max_value=len(pages), value=1, step=1)
            end_page = st.number_input("Enter end page", min_value=start_page, max_value=len(pages), value=start_page, step=1)

            combined_content = " ".join([page.extract_text() for page in pages[start_page - 1:end_page]])
            summary = summarize(combined_content)
            st.subheader("Summary")
            st.write(summary)
            st.button("Clear")

        elif page_selection == "Overall Summary":
            overall_content = " ".join([page.extract_text() for page in pages])
            summary = summarize(overall_content)
            st.subheader("Summary")
            st.write(summary)
            st.button("Clear")

        elif page_selection == "Question":
            question = st.text_input("Enter your question")
            answer = answer_question(pages, question)
            st.write("Answer:", answer)
            st.button("Clear")
