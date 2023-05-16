import datetime
import json
import os
import traceback
import asyncio
import redis
import dotenv
import requests
from flask import Flask, request, render_template, send_from_directory, jsonify
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import AzureChatOpenAI
import dotenv, os
from langchain.vectorstores.redis import Redis
from langchain.embeddings import OpenAIEmbeddings
import ast

from error import bad_request

# loading the .env file
dotenv.load_dotenv()

# Redirect PosixPath to WindowsPath on Windows
import platform

if platform.system() == "Windows":
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

def load_data():
    import os, json
    from langchain.document_loaders import UnstructuredMarkdownLoader
    from langchain.text_splitter import MarkdownTextSplitter
    from urllib.parse import quote
    
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores.redis import Redis

    excluded_files = ['SECURITY.md','SUPPORT.md', 'CODE_OF_CONDUCT.md', 'MIP-Readiness-Learning-Paths.md']
    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    # text_splitter = CharacterTextSplitter(
    #     separator="\n",
    #     chunk_size=1000, 
    #     chunk_overlap=200)

    def collect_document(source_path_list, replace_url_list):
        docs = []
        for source_path, replace_url in zip(source_path_list, replace_url_list):
            for root, folder, files in os.walk(source_path):
                for f in files:
                    if f[-2:]=='md' and f not in excluded_files:
                        # print(root,f)
                        try:
                            loader = UnstructuredMarkdownLoader(os.path.join(root, f))
                            data=loader.load()
                            
                            split_docs = text_splitter.create_documents([d.page_content for d in data])
                            for sd in split_docs:
                                id = os.path.join(root, f).replace(source_path,replace_url ).replace('\\', '/').replace('/csu-data_ai-main','').replace(' ','%20')
                                sd.metadata['source'] = id
                                docs.append(sd)
                        except Exception as e:
                            print(f'** Unable to load file {os.path.join(root,f)}')
                            print(e)
        return docs
    
    docs = collect_document(['./data/csu-data_ai-main'], 
                        ['https://github.com/customer-success-microsoft/csu-data_ai/blob/main'])
    print(f'Number of documents: {len(docs)}')

    redis_url = os.getenv("REDIS_URL") if os.getenv("DOCKER_REDIS_URL") is None else os.getenv("DOCKER_REDIS_URL")

    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDINGS_NAME"))
    vector_name='vbd_index'
    rds = Redis.from_documents([docs[0]], 
                            embeddings, 
                            redis_url=redis_url,  
                            index_name=vector_name)
    starting_idx = 1
    rds.add_documents(docs[starting_idx:])
    print('Redis Index created:', rds.index_name)
        

app = Flask(__name__)

# openai.api_type = "azure"
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_base = os.getenv("OPENAI_API_BASE")
# openai.api_version = os.getenv("OPENAI_API_VERSION") #openai api version m

with open("prompts/chat_combine_prompt.txt", "r") as f:
    chat_combine_template = f.read()

with open("prompts/chat_reduce_prompt.txt", "r") as f:
    chat_question_template = f.read()


messages_combine = [
    SystemMessagePromptTemplate.from_template(chat_combine_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
p_chat_combine = ChatPromptTemplate.from_messages(messages_combine)

messages_question = [
    SystemMessagePromptTemplate.from_template(chat_question_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
p_chat_question = ChatPromptTemplate.from_messages(messages_question)

redis_url = os.getenv("REDIS_URL") if os.getenv("DOCKER_REDIS_URL") is None else os.getenv("DOCKER_REDIS_URL")

vector_name='vbd_index'

embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDINGS_NAME"))


try:
    rds = Redis.from_existing_index(embeddings, redis_url=redis_url, index_name=vector_name)
    print('Index found.')
    
except:
    print('Index not found, creating new index')
    load_data()
    rds = Redis.from_existing_index(embeddings, redis_url=redis_url, index_name=vector_name)

print("Redis index loaded.")
async def async_generate(chain, question, chat_history):
    result = await chain.arun({"question": question, "chat_history": chat_history})
    return result


def run_async_chain(chain, question, chat_history):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = {}
    try:
        answer = loop.run_until_complete(async_generate(chain, question, chat_history))
    finally:
        loop.close()
    result["answer"] = answer
    return result

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/answer", methods=["POST"])
def api_answer():
    data = request.get_json()
    question = data["question"]
    if data['history'] != '[""]':
        history =  [(h1.replace("<br>", "\n"),h2.replace("<br>", "\n")) for h1,h2 in ast.literal_eval(data["history"])]
        # idx = 0
        # history = []
        # while idx < len(history_raw):
        #     history.extend([(history_raw[idx], history_raw[idx+1])])
        #     idx += 2
    else:
        history = []

    llm = AzureChatOpenAI(deployment_name=os.getenv("DEPLOYMENT_NAME"),
                        openai_api_version=os.getenv("OPENAI_API_VERSION"),
                            openai_api_key=os.getenv("OPENAI_API_KEY"),
                            openai_api_base=os.getenv("OPENAI_API_BASE"),
                            openai_api_type="azure",
                            temperature=0)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

    doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce", 
                                        question_prompt=p_chat_question,
                                        combine_prompt=p_chat_combine)
    chain = ConversationalRetrievalChain(
        retriever=rds.as_retriever(search_type="similarity"),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True
    )

    # use try and except  to check for exception
    try:

        answer = chain({"question": question, "chat_history": history})
        # history.append((question, result["answer"]))

        # print(result)
        result = {}
        # some formatting for the frontend
        result['answer'] = answer['answer']
        result['answer'] = result['answer'].replace("\\n", "\n")
        sources = set([s.metadata['source'] for s in answer["source_documents"]])
        result['sources'] = '<br>'.join(list(sources))
        history.append((question, result['answer']))
        result['chatHistory']=history
        # for s in sources:
        #     print("Source:     ",s)

        # try:
        #     result['answer'] = result['answer'].split("SOURCES:")[0]
        # except:
        #     pass

        return result
    except Exception as e:
        # print whole traceback
        traceback.print_exc()
        print(str(e))
        return bad_request(500, str(e))

# handling CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


if __name__ == "__main__":
    app.run(debug=True, port=5010)
