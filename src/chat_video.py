import streamlit as st
import whisper
from dotenv import load_dotenv
import pytube
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import os
import pytube
from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
from time import sleep
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain


@st.cache_data
def load_whisper_model():
    model = whisper.load_model("base")
    return model

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

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True
    )
    return conversation_chain

def summarizeMapReduce(data):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    text_splitter = CharacterTextSplitter(separator=" ", chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([data])
    map_prompt = """
        Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:
        """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    combine_prompt = """
        Write a concise summary of the following and give it in bullet points
        {text}
        CONCISE SUMMARY:
        """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    summary_chain = load_summarize_chain(llm=llm,
                                         chain_type='map_reduce',
                                         map_prompt=map_prompt_template,
                                         combine_prompt=combine_prompt_template
                                         )
    output = summary_chain.run(docs)
    return output

def main():
    load_dotenv()

    # Initialize OpenAI model
    client = OpenAI()

    if "youtube_text" not in st.session_state:
        st.session_state.youtube_text = []

    with st.sidebar:
        st.title('🤖💬 Video/Audio Chatbot')
        youtube_url = st.text_input("Add your YouTube URL")
        if st.button("Transcribe") and youtube_url:
            with st.spinner("Transcribing"):
                youtube_video = pytube.YouTube(youtube_url)
                streams = youtube_video.streams.filter(only_audio=True)
                stream = streams.first()
                stream.download(filename="youtube_audio.mp4")
                model = load_whisper_model()
                audio_file = open("youtube_audio.mp4", "rb")

                st.session_state.youtube_text = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file, 
                    response_format="text"
                    )

                # Chunk your texts
                text_chunks = get_text_chunks(st.session_state.youtube_text)

                # Add to pinecone and get vectorstore
                vectorstore = get_vectorestore(text_chunks)

                st.session_state.chain = get_conversation_chain(vectorstore)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            # Include entire history in current prompt to langchain
            if "summary" in prompt.lower() or "summarize" in prompt.lower():
                full_response = summarizeMapReduce(st.session_state.youtube_text)
            else:
                chat_prompt = "\n".join([m["content"] for m in st.session_state.messages])

                full_response = ""
                # Pass your chat_prompt to Langchain instead of individual messages
                response = st.session_state.chain({'question': chat_prompt})
                # Langchain's response includes the full history, so you need to extract only the latest response
                # Assuming the AI's message is always the last one
                full_response = response["answer"]
            # Create a typing effect by updating the display for each character
            typing_response = ""
            for character in full_response:
                typing_response += character
                message_placeholder.markdown(typing_response + "▌ ")
                sleep(0.02)  # delay to mimic typing, adjust the value as needed

        message_placeholder.markdown(typing_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        
if __name__ == '__main__':
    main()