import datetime
import json
import os
import traceback
import asyncio
import redis
import dotenv
import requests
from flask import Flask, request, render_template, send_from_directory, jsonify
from langchain.vectorstores.redis import Redis
from langchain.llms import AzureOpenAI

from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from werkzeug.utils import secure_filename
from AzureOpenAIUtil.AzureFormRecognizer import AzureFormRecognizerRead
import openai

from error import bad_request
\
# loading the .env file
dotenv.load_dotenv()

if os.getenv("LLM_NAME") is not None:
    llm_choice = os.getenv("LLM_NAME")
else:
    llm_choice = "openai_chat"

if os.getenv("EMBEDDINGS_NAME") is not None:
    embeddings_choice = os.getenv("EMBEDDINGS_NAME")
else:
    embeddings_choice = "text-embedding-ada-002"

embeddings = OpenAIEmbeddings(model=embeddings_choice)
# Redirect PosixPath to WindowsPath on Windows
import platform

if platform.system() == "Windows":
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

redis_url = os.getenv("REDIS_URL") if os.getenv("DOCKER_REDIS_URL") is None else os.getenv("DOCKER_REDIS_URL")
# load the prompts
with open("prompts/combine_prompt.txt", "r") as f:
    template = f.read()

with open("prompts/combine_prompt_hist.txt", "r") as f:
    template_hist = f.read()

with open("prompts/question_prompt.txt", "r") as f:
    template_quest = f.read()

with open("prompts/chat_combine_prompt.txt", "r") as f:
    chat_combine_template = f.read()

with open("prompts/chat_reduce_prompt.txt", "r") as f:
    chat_reduce_template = f.read()

if os.getenv("API_KEY") is not None:
    api_key_set = True
else:
    api_key_set = False


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER = "inputs"
app.config['EXTRACTED_FOLDER'] = EXTRACT_FOLDER = "extracted"
openai.api_type = "azure"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION") #openai api version m

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

# Add username and index name to local index file
def add_index_to_list(user_name, index_name):
    try:
        with open("index_list.json", "r") as f:
            index_list = json.load(f)
    except:
        index_list = {}
    if user_name not in index_list:
        index_list[user_name] = [index_name]
    else:
        if index_name not in index_list[user_name]:
            index_list[user_name].append(index_name)
    with open("index_list.json", "w") as f:
        json.dump(index_list, f)

# get list of indexes for a user
def get_index_list(user_name):
    with open("index_list.json", "r") as f:
        index_list = json.load(f)
    if user_name in index_list:
        for index_name in index_list[user_name]:
            try:
                redis = Redis.from_existing_index(embeddings, redis_url=redis_url, index_name=index_name)
            except:
                index_list[user_name].remove(index_name)
        index_list[user_name] = list(set(index_list[user_name]))
        with open("index_list.json", "w") as f:
            json.dump(index_list, f)
        return index_list[user_name]
    else:
        return []

@app.route("/")
def home():
    return render_template("index.html", api_key_set=api_key_set, llm_choice=llm_choice,
                           embeddings_choice=embeddings_choice)


@app.route("/api/answer", methods=["POST"])
def api_answer():
    data = request.get_json()
    question = data["question"]
    history = data["history"]
    print('-' * 5)
    # if not api_key_set:
    #     api_key = data["api_key"]
    # else:
    api_key = os.getenv("OPENAI_API_KEY")
    
    api_base = os.getenv("OPENAI_API_BASE")

    # use try and except  to check for exception
    try:
        # check if the vectorstore is set
        index_name = data["active_docs"].split('/')[-1]
        print(index_name)
        # vectorstore = "outputs/inputs/"
        # loading the index and the store and the prompt template
        # Note if you have used other embeddings than OpenAI, you need to change the embeddings

        # docsearch = FAISS.load_local(vectorstore, OpenAIEmbeddings(model=embeddings_choice))


        # create a prompt template
        if history:
            history = json.loads(history)
            template_temp = template_hist.replace("{historyquestion}", history[0]).replace("{historyanswer}",
                                                                                           history[1])
            c_prompt = PromptTemplate(input_variables=["summaries", "question"], template=template_temp,
                                      template_format="jinja2")
        else:
            c_prompt = PromptTemplate(input_variables=["summaries", "question"], template=template,
                                      template_format="jinja2")

        q_prompt = PromptTemplate(input_variables=["context", "question"], template=template_quest,
                                  template_format="jinja2")
        if llm_choice == "openai_chat":
            # llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4")
            llm = AzureChatOpenAI(openai_api_key=api_key,
                                  openai_api_base=api_base, 
                                  openai_api_type=os.getenv("OPENAI_API_TYPE"),
                                  openai_api_version=os.getenv("OPENAI_API_VERSION"),
                                  deployment_name=os.getenv("DEPLOYMENT_NAME"))
            messages_combine = [
                SystemMessagePromptTemplate.from_template(chat_combine_template),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
            p_chat_combine = ChatPromptTemplate.from_messages(messages_combine)
            messages_reduce = [
                SystemMessagePromptTemplate.from_template(chat_reduce_template),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
            p_chat_reduce = ChatPromptTemplate.from_messages(messages_reduce)

            question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
            rds = Redis.from_existing_index(embeddings, redis_url=redis_url, index_name=index_name)
            doc_chain = load_qa_chain(llm, chain_type="map_reduce", combine_prompt=p_chat_combine)
            chain = ConversationalRetrievalChain(
                retriever=rds.as_retriever(search_type="similarity"),
                question_generator=question_generator,
                combine_docs_chain=doc_chain,
            )
            chat_history = []
            result = chain({"question": question, "chat_history": chat_history})
 

        print(result)

        # some formatting for the frontend
        if "result" in result:
            result['answer'] = result['result']
        result['answer'] = result['answer'].replace("\\n", "\n")
        try:
            result['answer'] = result['answer'].split("SOURCES:")[0]
        except:
            pass

        return result
    except Exception as e:
        # print whole traceback
        traceback.print_exc()
        print(str(e))
        return bad_request(500, str(e))


@app.route("/api/docs_check", methods=["POST"])
def check_docs():
    data = request.get_json()
    index_name = data["docs"].split('/')[-1]
    try:
        rds = Redis.from_existing_index(embeddings, redis_url=redis_url, index_name=index_name)
        return {"status": 'exists'}
    except:
        # remove this index from the index list
        with open("index_list.json", "r") as f:
            index_list = json.load(f)
        user_name = 'local'
        if index_name not in index_list[user_name]:
            index_list[user_name].remove(index_name)
        with open("index_list.json", "w") as f:
            json.dump(index_list, f)
        return {"status": 'not_exists'}


@app.route('/api/combine', methods=['GET'])
def combined_json():
    user = 'local'
    """Provide json file with combined available indexes."""
    # get json from https://d3dg1063dc54p9.cloudfront.net/combined.json

    data = []
    # List indexes for user
    for index in get_index_list(user):
        data.append({
            "name": index,
            "language": '',
            "version": '',
            "description": index,
            "fullName": index,
            "date": '',
            "docLink": '',
            "model": embeddings_choice,
            "location": "local"
        })
    
    return jsonify(data)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload a file to get vectorized and indexed."""
    if 'user' not in request.form:
        return {"status": 'no user'}
    user = secure_filename(request.form['user'])
    if 'name' not in request.form:
        return {"status": 'no name'}
    job_name = secure_filename(request.form['name'])
    # check if the post request has the file part
    if 'file' not in request.files:
        print('No file part')
        return {"status": 'no file'}
    file = request.files['file']
    if file.filename == '':
        return {"status": 'no file name'}

    if file:
        filename = secure_filename(file.filename)
        # save dir
        save_dir = os.path.join(app.config['UPLOAD_FOLDER'], user, job_name)
        # create dir if not exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file.save(os.path.join(save_dir, filename))
        # task = ingest.delay('temp', [".rst", ".md", ".pdf", ".txt"], job_name, filename, user)

        azure_fr = AzureFormRecognizerRead()
        extracted_folder = os.path.join(app.config['EXTRACTED_FOLDER'], user, job_name)
        azure_fr.extract_files(save_dir, extracted_folder)

        # Extract text from JSON files from extracted folder
        from langchain.text_splitter import CharacterTextSplitter
        from langchain.schema import Document

        docs = []
        for file in os.listdir(extracted_folder):
            print('Loading file:', file)
            with open(os.path.join(extracted_folder, file)) as f:
                page_content= json.loads(f.read())
            docs.extend([Document(page_content = page['page_content'], metadata={'page_number': page['page_number']}) for page in page_content['content']])
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        processed_docs = text_splitter.split_documents(docs)

        # check if redis index already exists
        try:
            rds = Redis.from_existing_index(embeddings, redis_url=redis_url, index_name=job_name)
            starting_idx = 0
        except:
            rds = Redis.from_documents([processed_docs[0]], 
                                    embeddings, 
                                    redis_url=redis_url,  
                                    index_name=job_name)
            print('Redis Index created:', rds.index_name)
            starting_idx = 1
        add_index_to_list(user, job_name)
        # add documents to redis index
        rds.add_documents(processed_docs[starting_idx:])
        print('Redis Index updated:', rds.index_name)
        return {"status": 'ok'}
    else:
        return {"status": 'error'}


# @app.route('/api/task_status', methods=['GET'])
def task_status():
    raise NotImplementedError

# ### Backgound task api
@app.route('/api/upload_index', methods=['POST'])
def upload_index_files():
    raise NotImplementedError


@app.route('/api/download', methods=['get'])
def download_file():
    raise NotImplementedError


@app.route('/api/delete_old', methods=['get'])
def delete_old():
    raise NotImplementedError


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
