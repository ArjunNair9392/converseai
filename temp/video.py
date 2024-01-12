import streamlit as st
import whisper
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
import pytube
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from langchain_core.runnables import RunnableLambda
from langchain.document_loaders import ImageCaptionLoader
from langchain.docstore.document import Document
import os
import pytube
from openai import OpenAI
import librosa
from pydub import AudioSegment
import ffmpeg
import av
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone


@st.cache_data
def load_whisper_model():
    model = whisper.load_model("base")
    return model

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
    return conversation_chain

def get_vectorestore(chunks):
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_API_ENV")
    )
    index_name = "converseai"
    vectorstore = Pinecone.from_texts(texts=chunks, embedding=embeddings, index_name=index_name)
    return vectorstore

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(raw_text)
    return text_chunks

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with video")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    client = OpenAI()

    st.header("Chat with your video :video:")
    user_question = st.text_input("Ask a question about your document:") 
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your audio or video")
        youtube_url = st.text_input("YouTube URL")
        documents = []
        if youtube_url:
            youtube_video = pytube.YouTube(youtube_url)
            streams = youtube_video.streams.filter(only_audio=True)
            stream = streams.first()
            stream.download(filename="youtube_audio.mp4")
            model = load_whisper_model()
            audio_file = open("youtube_audio.mp4", "rb")
            # input_file = 'youtube_audio.mp4'
            # output_file = 'youtube_audio.wav'

            # ffmpeg.input(input_file).output(output_file, ar=16000).run()
            # audio_data, _ = librosa.load(output_file, sr=16000)

            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="text"
                )
            youtube_text = transcript

            # Create a Langchain document instance for the transcribed text
            youtube_document = Document(page_content=youtube_text, metadata={})
            documents.append(youtube_document)
            # st.write(youtube_text)

            # text chunking
            text_chunks_pdf = get_text_chunks(youtube_text)
            st.write(text_chunks_pdf)

            # create vector store
            vectorstore = get_vectorestore(text_chunks_pdf)

            # conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()