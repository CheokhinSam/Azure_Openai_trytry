# Import required libraries
from dotenv import load_dotenv
import os
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI  # Updated import
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
LLM_DEPLOYMENT = os.getenv("LLM_DEPLOYMENT")

# Validate that all required variables are set
if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, EMBEDDING_DEPLOYMENT, LLM_DEPLOYMENT]):
    raise ValueError("Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, EMBEDDING_DEPLOYMENT, and LLM_DEPLOYMENT in the .env file")

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=EMBEDDING_DEPLOYMENT,
    api_version="2023-05-15"
)

# Define a small set of documents
documents = [
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital and largest city of Germany.",
    "Tokyo is the capital of Japan and one of the most populous cities in the world.",
    "Ottawa is the capital city of Canada.",
    "Canberra is the capital city of Australia."
]

# Create a FAISS vector store from the documents
vectorstore = FAISS.from_texts(documents, embeddings)

# Set up the retriever (retrieve top 3 most relevant documents)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize Azure OpenAI Chat Model
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=LLM_DEPLOYMENT,  # Should point to a chat model like gpt-4o
    api_version="2023-05-15"  # Consider updating to "2024-02-15-preview" if needed
)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Combines retrieved documents into the prompt
    retriever=retriever
)

# Interactive query loop
print("Simple RAG Project with Azure OpenAI and LangChain")
while True:
    user_query = input("Ask a question (or type 'quit' to exit): ")
    if user_query.lower() == "quit":
        break
    response = qa_chain.invoke(user_query)["result"]  # Extract the result from the dictionary
    print(f"Response: {response}\n")