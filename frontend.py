import gradio as gr
import requests

def chat(query, chat_history):
    response = requests.post("https://smart-trivially-katydid.ngrok-free.app/query", json={"query": query, "chat_history": chat_history})
    return response.json().get("response", "Error: No response from backend")

chatbot = gr.Chatbot(avatar_images=["user.jpg", "bot.png"], height=600)
clear_but = gr.Button(value="Clear Chat History")
demo = gr.ChatInterface(fn=chat, title="Customized RAG Chatbot for Tenant FAQ", multimodal=False, retry_btn=None, undo_btn=None, clear_btn=clear_but, chatbot=chatbot)

if __name__ == '__main__':
    demo.launch(debug=True, share=True)