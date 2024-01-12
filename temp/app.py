import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import io
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from langchain.vectorstores import Pinecone
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from langchain_core.runnables import RunnableLambda
from langchain.document_loaders.csv_loader import CSVLoader
import pinecone
import os
import pandas as pd
from pandasai import SmartDataframe
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len
    )
    text_chunks = text_splitter.split_text(raw_text)
    return text_chunks

def chat_with_csv(df,prompt):
    llm = ChatOpenAI()
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    result = pandas_ai.chat(prompt)
    return result

def get_vectorestore(chunks):
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_API_ENV")
    )
    index_name = "converseai"
    vectorstore = Pinecone.from_texts(texts=chunks, embedding=embeddings, index_name=index_name)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    def fallback_response(inputs):
        return "I don't know."
    fallback_chain = RunnableLambda(fallback_response)

    chain_with_fallback = conversation_chain.with_fallbacks([fallback_chain])
    # chain_with_fallback = RunnableWithFallbacks(main_runnable=conversation_chain, fallbacks=[fallback_chain])
    return chain_with_fallback

def get_relevance(query, response):
    # Load the BERT model (this could be any model you choose)
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Get the embeddings
    query_embedding = model.encode([query])[0]
    response_embedding = model.encode([response])[0]

    # Calculate the cosine similarity
    similarity = cosine_similarity(
        query_embedding.reshape(1, -1),
        response_embedding.reshape(1, -1)
    )[0][0]

    print("Similarity:", similarity)

def handle_user_input(user_question):
    response = st.session_state.conversation.invoke({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    print("pdf:")
    get_relevance(user_question, response["answer"])


    # if any(keyword in response["answer"].lower() for keyword in ["don't know", "i'm sorry", "can't answer", "does not contain", "there is no information", "it is not possible to determine"]):
    data = pd.read_csv("titanic.csv")
    # # st.dataframe(data, use_container_width=True)
    result = chat_with_csv(data, user_question)
    print("csv:")
    print(str(result))
    get_relevance(user_question, str(result))
    # # st.success(result)
    # # st.write(result)
    # st.write(bot_template.replace(
    #                 "{{MSG}}", str(result)), unsafe_allow_html=True)
    # else:
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDF", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with your pdf :books:")
    user_question = st.text_input("Ask a question about your document:") 
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_file = st.file_uploader(
            "Upload your PDFs/csv here and click on 'Process'", accept_multiple_files=False, type=['pdf'])
        if st.button("Process"):
            with st.spinner("processing"):
                if uploaded_file is not None:
                    # Convert BytesIO objects to PyPDF2 PdfFileReader objects
                    # pdf_readers = [io.BytesIO(pdf.getvalue()) for pdf in uploaded_file]
                    pdf_readers = io.BytesIO(uploaded_file.getvalue())
                    
                    # get pdf text
                    raw_text = get_pdf_text(pdf_readers)
                    
                    # text chunking
                    text_chunks_pdf = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorestore(text_chunks_pdf)

                    # conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
        
if __name__ == '__main__':
    main()