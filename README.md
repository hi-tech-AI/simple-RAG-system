# Customized RAG Chatbot for Tenant FAQ

This project is a Retrieval-Augmented Generation (RAG) chatbot prototype that leverages various web scraping techniques, natural language processing (NLP), and a neural network model to provide answers based on scraped context.

## Overview

The primary steps involved in this project include:

1. **Web Scraping**: Extracts content from predefined URLs.
2. **Document Processing and Vectorization**: Uses natural language embeddings to process and store document chunks.
3. **Query Handling**: Retrieves the most relevant text segments to answer user queries.
4. **Chat Interface**: Provides an interactive chat interface using Gradio.

## Files and Directories Structure

- `main.py`: Main script containing all the logic for scraping, processing, and interaction with the chatbot.
- `content.txt`: File where the scraped content is stored.
- `customized_vector_db/`: Directory storing the vectorized documents for efficient retrieval.
- `README.md`: This readme file.

## Setting Up and Running

### Prerequisites

- Python 3.7+
- Required Python libraries (`requests`, `bs4`, `dotenv`, `gradio`, `langchain_community`, `langchain_core`, `langchain_openai`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourrepo/rag-chatbot.git
   cd rag-chatbot
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

### Running the Chatbot

1. To run the chatbot, execute the following command:
   ```bash
   python app.py
   ```

   This will launch a Gradio interface which can be accessed through a local web browser.

## Functions Explained

### Web Scraping

The `get_text` function scrapes text content from predefined URLs and saves it into `content.txt`. It extracts text within specific HTML tags such as 'p', 'h2', 'h3', 'li', and 'h4'.

### Creating Vector Database

The `create_vectordb` function loads the text data, splits them into chunks, and stores these chunks into a vector database using embeddings from OpenAI.

### Query Handling

The `rag_bot` function performs the core retrieval-augmented generation:
- It fetches the most similar documents to the query.
- Processes the query and context through a pre-trained language model to generate responses.

### Chat Interface

Utilizes Gradio to create a chat interface:
```python
chatbot = gr.Chatbot(avatar_images=["user.jpg", "bot.png"], height=600)
```
This interface allows users to interact with the chatbot conveniently.

## Customizing

To add new URLs for scraping or change existing ones, modify the `url` list in the `get_text` function. Ensure the URLs return status code `200`.

```python
url = [
    "https://new-url.com",
    ...
]
```

Remember to run `get_text()` and `create_vectordb()` again after updating the URLs to refresh the content and vector database.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.