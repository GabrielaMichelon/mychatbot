from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

# Path to the vector databse where the text is stored in chuncks
DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Sets up a retrieval-based QA chain using the provided language model, prompt, and database
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}), # return the top 2 most relevant results 
                                       return_source_documents=True, # return the parts of the source documents from which the answers were derived, alongside the answers themselves
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

# Loading the large language model (LLM), specified by its name and other parameters
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin", # "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama", # with 2.7 billion parameters, optimized for chat applications
        max_new_tokens = 512, # it specifies the maximum number of tokens (which can be words, parts of words, or punctuation marks) that the model is allowed to generate in a response to a given question/input 
        temperature = 0.5 # the temperature is a parameter that influences the randomness of the model's responses. A lower temperature (like 0.5) makes the model's responses more deterministic and less random
    )
    return llm

# QA Model Function: initializes the embeddings, loads the FAISS database, loads the LLM, sets up the QA prompt, and creates the QA chain
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Output function: takes a query, gets a QA result from the bot, and returns the response
#def final_result(query):
#    qa_result = qa_bot()
#    response = qa_result({'query': query})
#    return response

# Chainlit code - initializes the chatbot
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the chatbot...")
    await msg.send()
    msg.content = "Hi, Welcome to My chatbot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

#  Chainlit code - defines the behavior of the chatbot when it receives a message
    # cl.on_message is a decorator provided by the chainlit library to modify or extend the behavior of the main function
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()

