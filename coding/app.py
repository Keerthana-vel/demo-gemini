import os
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

load_dotenv()

api_key = os.environ.get('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not found.")

def get_file_paths(filenames):
    """Gets the absolute paths of multiple files in the same directory as the script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_paths = [os.path.join(script_dir, filename) for filename in filenames]
    return file_paths

# Read all pdf files and return text
file_paths = get_file_paths(['avr-a1h-owners-manual-en.pdf', 'avr-a1h-info-sheet-en.pdf'])

def file_read(file_paths):
    text = ""
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as pdf:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

def precompute_data(file_paths):
    raw_text = file_read(file_paths)
    text_chunks = get_text_chunks(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.load_local("faiss_index",embeddings, allow_dangerous_deserialization=True)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    You are a helpful assistant. Use the context provided to answer the question as naturally and conversationally as possible. If the context does not contain the answer, feel free to provide a plausible response based on general knowledge.

    Context: {context}

    Question: {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question"}]

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
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    return response


def main():
    st.set_page_config(
        page_title="AVR Chatbot",
        page_icon="🤖"
    )

    if not os.path.exists("faiss_index"):
        st.write("Precomputing data, please wait...")
        precompute_data(file_paths)
    else:
        st.write("Loading precomputed data...")

    st.title("Chat with AVR Assist 🤖")
    st.write("""Welcome to the chat! 
            Interested in AV Receiver!! 
            I'm ready for your questions!!!""")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question"}]
    
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
                    response = {"output_text": "Hello! Provide a question related to AV receiver!"}
                else:
                    response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
               # Ensure that response['output_text'] is handled correctly
                output_text = response.get('output_text', "")
            
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
