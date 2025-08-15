import os
import asyncio
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# ---------------------- FIX FOR NO EVENT LOOP IN THREAD ----------------------
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
# ------------------------------------------------------------------------------


load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not found.")
genai.configure(api_key=api_key)
models = genai.list_models()
for model in models:
    print(model.name)


def get_file_paths(filenames):
    """Gets the absolute paths of multiple files in the same directory as the script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_paths = [os.path.join(script_dir, filename) for filename in filenames]
    return file_paths


# Read all pdf files and return text
file_paths = get_file_paths([])

# Provide web links here
web_links = [
    "https://ez.analog.com/video/",
]


def file_read(file_paths):
    text = ""
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            with open(file_path, "rb") as pdf:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
    return text


# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings


# fetch data from the URL
def fetch_text_from_url(url):
    """Fetch and clean text from a webpage."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style tags
        for tag in soup(["script", "style"]):
            tag.extract()

        text = soup.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return text
    except Exception as e:
        return f"Error fetching {url}: {e}"


def precompute_data(file_paths, web_links):
    # 1. Read PDF text
    pdf_text = file_read(file_paths)

    # 2. Fetch all web link text
    web_text = ""
    for link in web_links:
        st.write(f"Fetching content from: {link}")
        web_text += fetch_text_from_url(link) + "\n"

    # 3. Merge into one big text block
    combined_text = pdf_text + "\n" + web_text

    # 4. Split into chunks
    text_chunks = get_text_chunks(combined_text)

    # 5. Create embeddings + FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )  # type: ignore
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    return vector_store


def get_conversational_chain():
    prompt_template = """
You are a helpful assistant. 
Answer ONLY from the provided context. 
If the answer is not in the context, respond with: "I couldn't find that in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",
        temperature=0.3,
        google_api_key=api_key,
    )
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question"}]


def handle_greeting(prompt):
    greetings = ["hi", "hello", "hey"]
    for greeting in greetings:
        if greeting in prompt.lower():
            return True
    return False


def user_input(user_question):
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )

    return response


def main():
    st.set_page_config(page_title="AVR Chatbot", page_icon="🤖")

    # if not os.path.exists("faiss_index"):
    #     st.write("Precomputing data, please wait...")
    #     precompute_data(file_paths, web_links)
    # else:
    #     st.write("Loading precomputed data...")
    precompute_data(file_paths, web_links)

    st.title("AVD7842")
    st.write("""Welcome to the chat!""")
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if handle_greeting(prompt):
                    response = {"output_text": "Hello! how can i help you?"}
                else:
                    response = user_input(prompt)
                placeholder = st.empty()
                full_response = ""
                # Ensure that response['output_text'] is handled correctly
                output_text = response.get("output_text", "")

                if isinstance(output_text, str):
                    # If output_text is a string, iterate through its characters
                    for item in output_text:
                        full_response += item
                        placeholder.markdown(full_response)
                elif isinstance(output_text, list):
                    # If output_text is a list, iterate through its elements
                    for item in output_text:
                        full_response += item
                        placeholder.markdown(full_response)
                else:
                    # If output_text is neither a string nor a list, handle it as an error or unexpected format
                    full_response = "Unexpected response format."
                    placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
