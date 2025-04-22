import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env
load_dotenv()

## Loading GROQ API KEY
groq_api_key = os.environ['GROQ_API_KEY']

# Define persistent directory for vector storage
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector store with the embeddings function
try:
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
except Exception as e:
    print(f"Error loading Chroma vector store: {e}")
    exit()

# Create a retriever from the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

# Initialize the LLM with GPT-4 model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-70b-8192")

# Contextualizing question prompt for history-based reformulation
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, "
    "formulate a standalone question that can be understood without chat history. "
    "Do not answer the question, just reformulate it if needed."
)

# Create a prompt template for contextualizing the question
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever that uses LLM for query reformulation
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# System prompt for question answering
qa_system_prompt = (
    "You are an assistant for answering questions. Use the following context to answer the "
    "question. If the answer is unknown, say 'I don't know'. Provide a concise answer "
    "in three sentences or fewer.\n\n{context}"
)

# Create a prompt template for the QA task
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Chain to combine the retrieved context for question answering
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Final RAG chain that combines the history-aware retriever with the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Chat function to handle user interaction
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Stores the chat history as a sequence of messages

    while True:
        try:
            query = input("You: ")
            if query.lower() == "exit":
                break
            # Process the query through the RAG chain
            result = rag_chain.invoke({"input": query, "chat_history": chat_history})
            # Extract AI's answer
            ai_response = result.get('answer', 'No answer available.')
            print(f"AI: {ai_response}")
            # Update chat history with user query and AI response
            chat_history.append(HumanMessage(content=query))
            chat_history.append(SystemMessage(content=ai_response))
        except Exception as e:
            print(f"An error occurred: {e}")
            continue


# Main function to launch the chatbot
if __name__ == "__main__":
    continual_chat()