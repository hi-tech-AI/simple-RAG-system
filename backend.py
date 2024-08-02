from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4-1106-preview", api_key=OPENAI_API_KEY)

vectordb_path = './customized_vector_db'

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query')
    chat_history = data.get('chat_history', [])

    response = rag_bot(user_query, chat_history)
    return jsonify({'response': response})

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

if __name__ == '__main__':
    app.run(port=5002, debug=True)