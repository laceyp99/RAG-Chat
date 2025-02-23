# RAG Chatbot 🤖

This application is a **Retrieval Augmented Generation (RAG) chatbot** built with LangChain and Gradio. It processes uploaded documents, splits them into chunks, and builds a vector store so that user prompts are answered in context using OpenAI's GPT-4o-mini model.

![image](https://github.com/user-attachments/assets/fb6435aa-9d11-405f-a95f-71208751ac49)

## Supported File Types 📂

Currently, the application supports the following file types for document uploads:

- **PDF** (`.pdf`)
- **Text** (`.txt`)
- **Markdown** (`.md`)
- **JSON** (`.json`)
- **CSV** (`.csv`)


## Features 🚀

- **Document Loading:** Upload files via the Gradio interface and load them into the system.
- **Vector Store & Embeddings:** Uses Chroma and OpenAI embeddings to create a searchable document store.
- **RAG Chain Construction:** Builds a chain that retrieves context from loaded documents and answers user queries using a language model.
- **Adjustable Generation Temperature:** Dynamically control the generation temperature for the language model to vary response creativity.
- **Advanced Parameters:** Fine-tune document processing by adjusting chunk size, chunk overlap, and retrieval k via an advanced parameters accordion. This keeps the interface simple for basic users while providing additional control for advanced users.
- **User Interface:** A streamlined Gradio-based interface for document uploads and chatbot interactions.

## Setup and Installation 🛠️

1. **Clone the Repository:**

   ```shell
   git clone https://github.com/laceyp99/RAG_Chat.git
   cd RAG-Chat
   ```

3. **Install Dependencies:**
    To ensure that you have all the necessary python libraries and packages for this application, please run:
    ```shell
    pip install gradio python-dotenv langchain langchain_openai langchain_community bs4 chromadb cryptography pypdf
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
    * Upload supported document types (PDF, TXT, MD, JSON, CSV) 📄
    * Build the RAG chain ⚙️
    * Chat with the RAG bot using the loaded documents 💬

## Files Overview 🗂️
* **main.py**: Contains core logic for document loading, text splitting, vector store creation, and RAG chain construction.
* **app.py**: Defines the Gradio interface for document uploads and chatbot interactions.

## Notes 📌
- **Advanced Parameters Warning:**  
  - Adjusting chunk size and overlap can affect processing times and memory usage. Smaller chunk sizes or larger overlaps may lead to a significantly higher number of splits. ⚠️
  - Lower retrieval k values might reduce context for query responses, while higher values could slow down search performance.
- **Generation Temperature:**  
  - Changing the generation temperature alters the response creativity. Higher values might yield less predictable or overly creative responses. ⚠️
- **Document Requirements:**  
  - Ensure documents are well-formatted. PDFs should not be password-protected, and JSON files must be valid to load correctly.
- **User Control:**  
  - Default values are set to simplify basic usage; advanced users can tailor these settings for specific needs.

## Future Enhancements 🔮
- **Web URL Support:**  
  - Integrate the ability to scrape webpage content from supplied URLs and include it in the document pool.
- **Expanded File Type Support:**  
  - Add support for additional document types, such as DOCX, to broaden the scope of data that can be processed.
- **UI Enhancements:**  
  - Implement tooltips or inline documentation within the Gradio interface for advanced parameters and generation temperature, aiding user understanding.
- **Vector Store Persistence:**  
    - Enable saving and loading of the vector store across sessions for a more tailored and efficient experience.
