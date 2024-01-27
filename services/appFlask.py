import os

import PyPDF2
from flask import *
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from langchain.vectorstores import Pinecone
from langchain_core.runnables import RunnableLambda
import pinecone
import os
import pandas as pd
from pandasai import SmartDataframe
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ChatBot(object):

    def __init__(self, app, savedFilePath, **configs):
        self.app = app
        self.configs(**configs)
        self.savedFilePath = savedFilePath

    def configs(self, **configs):
        for config, value in configs:
            self.app.config[config.upper()] = value

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None, methods=['GET'], *args, **kwargs):
        self.app.add_url_rule(endpoint, endpoint_name, handler, methods=methods, *args, **kwargs)

    def run(self, **kwargs):
        self.app.run(**kwargs)

flask_app = Flask(__name__)

app = ChatBot(flask_app, '')


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

def main():
    return render_template("index.html")

def success():
    if request.method == 'POST':
        f = request.files['file']
        global savedFilePath
        app.savedFilePath = os.path.join('../docs', f.filename)
        f.save(app.savedFilePath)
        print(f.filename)
        return render_template("success.html", name=f.filename)

def get_vectorestore(chunks):
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_API_ENV")
    )
    index_name = "converseai"
    vectorstore = Pinecone.from_texts(texts=chunks, embedding=embeddings, index_name=index_name)
    return vectorstore

def process():
    if request.method == 'POST':
        from io import BytesIO
        print(app.savedFilePath)
        with open(app.savedFilePath, "rb") as fh:
            pdfReader = PyPDF2.PdfReader(fh)

            # printing number of pages in pdf file
            print(len(pdfReader.pages))
            raw_text = get_pdf_text(fh)
            print(raw_text)
            # text chunking
            text_chunks_pdf = get_text_chunks(raw_text)

            # create vector store
            # vectorstore = get_vectorestore(text_chunks_pdf)
            #
            # # conversation chain
            # st.session_state.conversation = get_conversation_chain(vectorstore)
        return jsonify(
            process="success"
        )

app.add_endpoint('/', 'main', main, methods=['GET'])
app.add_endpoint('/success', 'success', success, methods=['POST'])
app.add_endpoint('/process', 'process', process, methods=['POST'])



# @app.route('/success', methods=['POST'])
# def success():
#     if request.method == 'POST':
#         f = request.files['file']
#         global savedFilePath
#         savedFilePath = os.path.join('docs',f.filename)
#         f.save(savedFilePath)
#
#         return render_template("success.html", name=f.filename)
#
# @app.route('/process', methods=['POST'])
# def process():
#     if request.method == 'POST':
#         from io import BytesIO
#         with open(savedFilePath, "rb") as fh:
#             buf = BytesIO(fh.read())
#         return jsonify(
#             process="success"
#         )

# @app.route('/query', methods=['POST'])
# def handleUserQuery():
#     if request.method == 'POST':
#         request_data = request.get_json()
#         question = request_data['question']
#         print(question)
#         return jsonify(
#             answer="sample answer"
#         )

if __name__ == '__main__':
    app.run(debug=True)