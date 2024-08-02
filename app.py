import requests
from bs4 import BeautifulSoup

def get_text():
    url = [
        "https://www.hendersonproperties.com/property-management-services/faqs/",
        "https://www.shoreagents.com/property-management-maintenance-schedule/#:~:text=Property%20maintenance%20is%20an%20ongoing,various%20legal%20and%20regulatory%20standards",
        "https://fitsmallbusiness.com/rental-property-maintenance-checklist/",
        "https://www.avail.co/education/guides/complete-guide-to-rental-property-maintenance/preventative-maintenance-checklist",
        "https://propertymanagementofsoutherncalifornia.com/tenant-faqs/",
        "https://www.forbes.com/sites/forbesbusinesscouncil/2024/02/01/determining-maintenance-costs-for-rental-properties/#",
        "https://www.secondnature.com/blog/property-management-maintenance?hs_amp=true",
        "https://www.thebalancemoney.com/what-are-maintenance-expenses-5215132#:~:text=Maintenance%20expenses%20are%20part%20of,home's%20heating%20and%20cooling%20systems",
        "https://nobleproperties.info/rental-property-maintenance-faq-chula-vista/"
        ]

    tag_list = ['p', 'h2', 'h3', 'li', 'h4']
    text = ""

    for link in url:
        response = requests.get(link)
        for tag in tag_list:
            if response.status_code == 200:
                page_content = response.content
                soup = BeautifulSoup(page_content, 'html.parser')
                
                try:
                    tags = soup.find_all(tag)
                    for item in tags:
                        text += item.get_text()
                        print(item.get_text())
                        with open("content.txt", "a") as data:
                            data.write(f"{item.get_text()}\n")
                except:
                    pass
            else:
                print("Failed to retrieve the webpage. Status code:", response.status_code)

from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4-1106-preview", api_key=OPENAI_API_KEY)

vectordb_path = './customized_vector_db'
uploaded_files = ['content.txt']
vectorstore = None

def create_vectordb():
    global vectorstore
    for file in uploaded_files:
        loader = TextLoader(file)
        data = loader.load()
        texts = text_splitter.split_documents(data)

        if vectorstore is None:
            vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=os.path.join(vectordb_path))
        else:
            vectorstore.add_documents(texts)

def rag_bot(query, chat_history):
    print(f"Received query: {query}")

    template = """Please answer to human's input based on context. If the input is not mentioned in context, output something like 'I don't know'.
    Context: {context}
    Human: {human_input}
    Your Response as Chatbot:"""

    prompt_s = PromptTemplate(
        input_variables=["human_input", "context"],
        template=template
    )

    # Initialize vector store
    vectorstore = Chroma(persist_directory=os.path.join(vectordb_path), embedding_function=embeddings)

    docs = vectorstore.similarity_search(query)

    stuff_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_s)

    output = stuff_chain({"input_documents": docs, "human_input": query}, return_only_outputs=False)

    final_answer = output["output_text"]
    print(f"Final Answer ---> {final_answer}")

    return final_answer

def chat(query, chat_history):
    response = rag_bot(query, chat_history)
    # chat_history.append((query, response))
    return response

chatbot = gr.Chatbot(avatar_images=["user.jpg", "bot.png"], height=600)
clear_but = gr.Button(value="Clear Chat History")
demo = gr.ChatInterface(fn=chat, title="Customized RAG Chatbot for Tenant FAQ", multimodal=False, retry_btn=None, undo_btn=None, clear_btn=clear_but, chatbot=chatbot)

if __name__ == '__main__':
    # get_text()
    # create_vectordb()
    demo.launch(debug=True, share=True)