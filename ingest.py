from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

# The path to the directory where the PDF documents are stored
DATA_PATH = 'data/'
# The path to store the FAISS database
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database from PDF documents, which can then be used for question-answering
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader) # For loading PDF documents from a directory

    documents = loader.load()
    # To split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Generates embeddings models from Hugging Face for the chunks of texts
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Creates a FAISS database from these embeddings and saves it to DB_FAISS_PATH
    # FAISS is a database for efficient similarity search and clustering of dense vectors
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

