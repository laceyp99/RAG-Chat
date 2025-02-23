from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader, JSONLoader, CSVLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load the API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI API Clients
llm = ChatOpenAI(api_key=api_key, model_name="gpt-4o-mini", temperature=0)
embedding_model = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-large")

# Global variable to hold the RAG chain
rag_chain = None

# Format retrieved documents by joining their page_content
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_documents(files, web_urls):
    all_docs = []
    # Process web URLs: supports comma or newline separated input.
    if web_urls is not None:
        if isinstance(web_urls, str):
            urls = [url.strip() for url in web_urls.replace(",", "\n").splitlines() if url.strip()]
        else:
            urls = []
        if urls:
            print(f"Web URLs: {urls}")
            web_loader = WebBaseLoader(
                web_paths=urls,
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                ),
            )
            web_docs = web_loader.load()
            print(f"Web docs: {web_docs}")
            all_docs.extend(web_docs)

    # Process file uploads. Gradio file uploads are provided as dictionaries.
    if files is not None:
        for file in files:
            if file is None:
                continue

            file_ext = os.path.splitext(file)[1].lower()
            # Choose the loader based on the file extension.
            if file_ext == ".pdf":
                loader = PyPDFLoader(file_path=file)
            elif file_ext in [".txt", ".md"]:
                loader = TextLoader(file_path=file)
            elif file_ext in [".json"]:
                loader = JSONLoader(file_path=file)
            elif file_ext in [".csv"]:
                loader = CSVLoader(file_path=file)
            else:
                print(f"File type {file_ext} not supported. Skipping {file}.")
                continue

            file_docs = loader.load()
            all_docs.extend(file_docs)
            
    return all_docs

def build_chain(files):
    """
    Loads and processes documents from various loaders,
    builds a vectorstore and constructs the RAG chain.
    Returns the built rag_chain.
    """
    global rag_chain
    
    # Load documents from the various sources
    docs = load_documents(files=files, web_urls=None) # web_urls
    print(f"docs: {docs}")
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print(f"Number of splits: {len(splits)}")
    print(splits)
    if not splits:
        return "No documents loaded."
    # Create a vectorstore from the embedded chunks
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    retriever = vectorstore.as_retriever()
    
    # Pull the prompt template
    prompt = hub.pull("rlm/rag-prompt")
    
    # Build the chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return "Documents loaded successfully."

def process_query(query):
    """
    Action to be triggered when the user submits a question.
    It uses the global rag_chain, if available, to generate a response.
    """
    if rag_chain is None:
        return "Please load documents first."
    return rag_chain.invoke(query)

# Main function
# if __name__ == "__main__":
    # Load the documents
    # Then process queries
    # pass
