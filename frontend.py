import gradio as gr
import requests

def chat(query):
    response = requests.post("https://smart-trivially-katydid.ngrok-free.app/query",
                             headers={"Content-Type": "application/json",
                                      "X-API-KEY": "e3876688bb83c070232a5df305f92eeb"},
                             json={"query": query})
    return response.json().get("response", "Error: No response from backend")

chatbot = gr.Chatbot(avatar_images=["user.jpg", "bot.png"], height=600)
clear_but = gr.Button(value="Clear Chat History")
demo = gr.ChatInterface(fn=chat, title="Customized RAG Chatbot for Tenant FAQ", multimodal=False, retry_btn=None, undo_btn=None, clear_btn=clear_but, chatbot=chatbot)

if __name__ == '__main__':
    demo.launch(debug=True, share=True)