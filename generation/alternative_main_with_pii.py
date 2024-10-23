import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import spacy

load_dotenv()

DATA_PATH = 'Hackathon Planning 2024.pdf'

current_dir = os.path.dirname(os.path.abspath(__file__))

def censor_pii(documents):
    """
    Censors personally identifiable information (PII) from a list of documents.

    Args:
        documents (list): A list of documents, where each document is an object with a 'page_content' attribute containing the text to be censored.

    Returns:
        list: A list of censored documents, where each document is a string with PII replaced with '[REDACTED]'.

    Notes:
        This function uses the SpaCy library to identify entities in the text, and censors the following types of PII:
            - PERSON: Names of individuals
            - EMAIL: Email addresses
            - PHONE: Phone numbers
    """
    nlp = spacy.load("en_core_web_sm")
    for doc in documents:
        text = doc.page_content
        doc_spacy = nlp(text)
        for ent in doc_spacy.ents:
            if ent.label_ in ["PERSON", "EMAIL", "PHONE"]:
                text = text.replace(ent.text, "[REDACTED]")
        doc.page_content = text
    return documents

def create_vector_store(data_path, text_splitter_type, text_splitter_kwargs, embedding_model_type, embedding_model_kwargs):
    """
    Creates a vector store from a given data path and stores it in a persistent directory.

    Args:
        data_path (str): The path to the data file (e.g., a PDF file).
        text_splitter_type (callable): A callable that splits the text into chunks.
        text_splitter_kwargs (dict): Keyword arguments for the text splitter.
        embedding_model_type (callable): A callable that creates an embedding model.
        embedding_model_kwargs (dict): Keyword arguments for the embedding model.

    Returns:
        Chroma: The created vector store.

    Raises:
        FileNotFoundError: If the file at the specified data path does not exist.

    Notes:
        If the vector store already exists in the persistent directory, it will be loaded instead of recreated.
    """
    file_path = os.path.join(current_dir, data_path)
    persistent_directory = os.path.join(current_dir, "db", "chroma_db")
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store ---")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        documents = censor_pii(documents)
        text_splitter = text_splitter_type(**text_splitter_kwargs)
        docs = text_splitter.split_documents(documents)
        embeddings = embedding_model_type(**embedding_model_kwargs)
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store ---")
        return db
    else:
        print(f"Vector store already exists. No need to initialize.")
        embeddings = embedding_model_type(**embedding_model_kwargs)
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        return db

def query_vector_store(db, query, search_type, search_kwargs):
    """
    Queries the vector store to retrieve relevant documents based on the input query.

    Args:
        db (Chroma): The vector store to query.
        query (str): The input query to search for in the vector store.
        search_type (str): The type of search to perform (e.g., "dense", "exact").
        search_kwargs (dict): Additional keyword arguments for the search function.

    Returns:
        str: A combined input string that includes the original query and the relevant documents.

    Notes:
        This function uses the vector store's retriever to search for relevant documents based on the input query.
        The resulting documents are then combined with the original query to create a new input string.
    """
    retriever = db.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs, 
    )
    relevant_docs = retriever.invoke(query)
    combined_input = (
        "Here are some documents that might help answer the question: "
        + query
        + "\n\nRelevant Documents:\n"
        + "\n\n".join([doc.page_content for doc in relevant_docs])
        + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
    )
    return combined_input

def generate_response (llm_type, llm_kwargs, combined_input):
    """
    Generates a response to the input query using a large language model (LLM).

    Args:
        llm_type (callable): A callable that creates the llm. 
        llm_kwargs (dict): Keyword arguments for the llm.
        combined_input (str): The input string that includes the original query and relevant documents.

    Returns:
        str: The generated response to the input query.

    Notes:
        This function uses the LLM to generate a response based on the input string.
        The response is generated by passing the input string to the LLM and retrieving the output.
    """
    llm = llm_type(**llm_kwargs)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]
    result = llm.invoke(messages)
    return result.content

# Sample usage that uses recursivecharactertextsplitter, huggingfaceembeddings, and chatgooglegenerativeai (gemini)

db = create_vector_store(DATA_PATH, RecursiveCharacterTextSplitter, {'chunk_size':1000, 'chunk_overlap':500}, HuggingFaceEmbeddings, {"model_name": "sentence-transformers/all-mpnet-base-v2"})
combined_input = query_vector_store(db, "What is the document about?", "mmr", {"k": 3, "fetch_k": 20, "lambda_mult": 0.5})
response = generate_response(ChatGoogleGenerativeAI, {"model": "gemini-1.5-flash"}, combined_input)
print(response)
