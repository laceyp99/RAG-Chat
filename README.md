# RAG Chatbot

This application is a Retrieval Augmented Generation (RAG) chatbot built with LangChain and Gradio. It processes uploaded documents, splits them into chunks, and builds a vector store so that user prompts are answered in context using OpenAI's Chat model.

## Supported File Types

Currently, the application supports the following file types for document uploads:

- **PDF** (`.pdf`)
- **Text** (`.txt`)
- **Markdown** (`.md`)
- **JSON** (`.json`)
- **CSV** (`.csv`)

## Future Enhancements

- **Web URL Support:** Future versions will allow users to input web URLs to scrape webpage content and include it as part of the document pool.

## Features

- **Document Loading:** Upload files via the Gradio interface and load them into the system.
- **Vector Store & Embeddings:** Uses Chroma and OpenAI embeddings to create a searchable document store.
- **RAG Chain Construction:** Builds a chain that retrieves context from loaded documents and answers user queries using a language model.
- **User Interface:** Gradio-based interface to load documents and chat with the RAG bot.

## Setup and Installation

1. **Clone the Repository:**

   ```shell
   git clone https://github.com/laceyp99/RAG_Chat.git
   cd RAG_Chat
   ```

3. **Install Dependencies:**
    To ensure that you have all the necessary python libraries and packages for this application, please run:
    ```shell
    pip install gradio python-dotenv langchain langchain_openai langchain_community bs4
    ```

4. **Set Up Environment Variables:**
    Create a .env file in the project root directory and add your OpenAI API key:
    ```shell
    OPENAI_API_KEY=your_openai_api_key_here
    ```

5. **Run the Application:**
    Start the Gradio interface by running:
    ```shell
    python app.py
    ```

    This will launch a local server where you can:
    * Upload supported document types (PDF, TXT, MD, JSON, CSV)
    * Build the RAG chain
    * Chat with the RAG bot using the loaded documents

## Files Overview
* **main.py**: Contains core logic for document loading, text splitting, vector store creation, and RAG chain construction.
* **app.py**: Defines the Gradio interface for document uploads and chatbot interactions.

## Notes
* For optimal performance, ensure your documents are well-formatted and that PDFs are not password-protected.
* The current version only supports the listed file types. Web URL scraping will be available in future updates.