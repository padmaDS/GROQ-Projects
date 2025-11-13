### https://github.com/siddharth-Kharche/RAG-Chatbot-with-Memory-/blob/main/app.py

"""
RAG Chatbot with Memory using Groq LLM and Hugging Face Embeddings
Built with latest versions of Chainlit, LangChain, and Groq
"""

from typing import List
import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

PDF_FOLDER_PATH = "data"  # Your PDF documents folder
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Configuration
GROQ_MODEL = "llama-3.3-70b-versatile"  # Latest Groq model (Nov 2025)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Open-source embeddings

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TEMPERATURE = 0

# ============================================================================
# Document Processing Functions
# ============================================================================

def load_and_process_pdfs(pdf_folder_path: str) -> List[Document]:
    """
    Load and process PDF documents from a folder.
    
    Args:
        pdf_folder_path: Path to folder containing PDF files
        
    Returns:
        List of processed document chunks
    """
    documents = []
    
    # Check if folder exists
    if not os.path.exists(pdf_folder_path):
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder_path}")
    
    # Load all PDF files
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_folder_path}")
    
    print(f"Loading {len(pdf_files)} PDF files...")
    
    for file in pdf_files:
        pdf_path = os.path.join(pdf_folder_path, file)
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            print(f"✓ Loaded: {file}")
        except Exception as e:
            print(f"✗ Error loading {file}: {str(e)}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"✓ Created {len(splits)} document chunks")
    
    return splits


def initialize_vectorstore(splits: List[Document], embeddings_model) -> FAISS:
    """
    Initialize FAISS vector store with document chunks.
    
    Args:
        splits: List of document chunks
        embeddings_model: Embedding model instance
        
    Returns:
        FAISS vector store
    """
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(
        documents=splits, 
        embedding=embeddings_model
    )
    print("✓ Vector store created successfully")
    return vectorstore


# ============================================================================
# Initialize Models and Vector Store
# ============================================================================

print("=" * 60)
print("Initializing RAG Chatbot...")
print("=" * 60)

# Initialize embeddings model (Hugging Face - open source)
print("\n1. Loading embeddings model...")
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
    encode_kwargs={'normalize_embeddings': True}
)
print(f"✓ Loaded: {EMBEDDING_MODEL}")

# Load and process PDFs
print(f"\n2. Processing PDFs from: {PDF_FOLDER_PATH}")
splits = load_and_process_pdfs(PDF_FOLDER_PATH)

# Initialize vector store
print("\n3. Building vector database...")
vectorstore = initialize_vectorstore(splits, embeddings_model)

# Initialize Groq LLM (latest version)
print("\n4. Initializing Groq LLM...")
model = ChatGroq(
    model=GROQ_MODEL,
    temperature=TEMPERATURE,
    groq_api_key=GROQ_API_KEY,
    max_tokens=8192,  # Maximum context for response
)
print(f"✓ Model: {GROQ_MODEL}")

print("\n" + "=" * 60)
print("✓ Initialization Complete!")
print("=" * 60 + "\n")


# ============================================================================
# Chainlit Event Handlers
# ============================================================================

@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chat session when user connects.
    Sets up retriever, memory, and conversational chain.
    """
    
    # Send welcome message
    await cl.Message(
        content="🚀 **Welcome to Your RAG Expert Assistant!**\n\n"
                f"Powered by **Groq** ({GROQ_MODEL}) and **Hugging Face** embeddings.\n\n"
                "📚 I have access to your regulatory documents and can answer questions with memory of our conversation.\n\n"
                "💬 Ask me anything!"
    ).send()
    
    # Create retriever from vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve top 4 relevant chunks
    )
    
    # Initialize chat history
    message_history = ChatMessageHistory()
    
    # Create conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    
    # Create conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        chain_type="stuff",  # Stuff all context into prompt
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    
    # Store chain in user session
    cl.user_session.set("chain", chain)
    print(f"✓ New chat session started")


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages from users.
    Processes query through RAG chain and returns answer with sources.
    
    Args:
        message: User's message object from Chainlit
    """
    
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")
    
    if not chain:
        await cl.Message(
            content="❌ Error: Chat session not initialized. Please refresh the page."
        ).send()
        return
    
    # Create callback handler for async operations
    cb = cl.AsyncLangchainCallbackHandler()
    
    # Send thinking indicator
    thinking_msg = cl.Message(content="🤔 Thinking...")
    await thinking_msg.send()
    
    try:
        # Get response from the chain
        res = await chain.acall(
            message.content, 
            callbacks=[cb]
        )
        
        answer = res["answer"]
        source_documents = res["source_documents"]
        
        # Remove thinking message
        await thinking_msg.remove()
        
        text_elements = []
        
        # Process source documents
        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx + 1}"
                
                # Extract metadata
                page = source_doc.metadata.get('page', 'N/A')
                source_file = source_doc.metadata.get('source', 'Unknown')
                
                # Create text element for each source with metadata
                text_elements.append(
                    cl.Text(
                        content=f"**Page {page}** | {os.path.basename(source_file)}\n\n{source_doc.page_content}",
                        name=source_name,
                        display="side"  # Display in sidebar
                    )
                )
            
            source_names = [text_el.name for text_el in text_elements]
            
            # Append sources to answer
            if source_names:
                answer += f"\n\n---\n📚 **Sources:** {', '.join(source_names)}"
                answer += f"\n\n*Click on source names to view the relevant document excerpts.*"
            else:
                answer += "\n\n---\n❌ No sources found"
        else:
            answer += "\n\n---\n⚠️ Answer generated without specific document context"
        
        # Send response with source documents
        await cl.Message(
            content=answer, 
            elements=text_elements
        ).send()
        
    except Exception as e:
        # Remove thinking message on error
        await thinking_msg.remove()
        
        # Send error message
        await cl.Message(
            content=f"❌ **Error processing your request:**\n\n``````\n\nPlease try again or rephrase your question."
        ).send()
        print(f"Error: {str(e)}")


@cl.on_chat_end
async def on_chat_end():
    """
    Clean up when chat session ends.
    """
    print("✓ Chat session ended")


# ============================================================================
# Run Instructions
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("To run this application, use:")
    print("  chainlit run app.py -w")
    print("=" * 60 + "\n")