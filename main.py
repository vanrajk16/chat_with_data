import os
import datetime
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS


load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']

current_date = datetime.datetime.now().date()
llm_name = "gpt-3.5-turbo-0301" if current_date < datetime.date(2023, 9, 2) else "gpt-3.5-turbo"
embedding = OpenAIEmbeddings()
persist_directory = 'docs/chroma/'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
llm = ChatOpenAI(model_name=llm_name, temperature=0)

template = """Use the following pieces of context which is provided only to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.and use only that data context which is provided do not use any other data of yours. and also consider privous answer of question if the prompt is reletional. and just give the answer if you know only and in the context too, if not then say i don`t know. don`t try to give answer from out of the context.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

app = Flask(__name__)
CORS(app, supports_credentials=True)  

@app.route('/introduce', methods=['POST'])
def introduce():
    introduction = "Hello! I'm Your Virtual Assistent, . How can I assist you with My Orgenization?"
    return jsonify({'answer': introduction})

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data['question']

    result = qa_chain({"query": question})

    return jsonify(result["result"])

if __name__ == '__main__':
    app.run(debug=True)
