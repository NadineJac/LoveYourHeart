
import os

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import streamlit as st


# llm
model = "llama-3.3-70b-versatile"

llm = Groq(
    model=model,
    token=st.secrets["GROQ_API_KEY"], # when you're running local
)


# embeddings
embedding_model = "danielheinz/e5-base-sts-en-de"#"sentence-transformers/all-MiniLM-L6-v2"
embeddings_folder = "../content/embedding_model/"

embeddings = HuggingFaceEmbedding(
    model_name=embedding_model,
    cache_folder=embeddings_folder,
)

# load Vector Database
# allow_dangerous_deserialization is needed. Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on your machine
storage_context = StorageContext.from_defaults(persist_dir="../content/vector_index")
vector_index = load_index_from_storage(storage_context, embed_model=embeddings)

# retriever
retriever = vector_index.as_retriever(similarity_top_k=2)

# prompt
prefix_messages = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content='''
        Context: You are a friendly, supportive health assistant helping adults understand heart risk based on self-reported lifestyle factors (smoking, diet, exercise). You can reference clinical risk factors (blood pressure, cholesterol, family history) and lifestyle changes. Risk comes from an external ML model as “low,” “medium,” or “high.”
        Objective: Explain lifestyle risk factors clearly and briefly, provide context about clinical factors, interpret risk categories, encourage achievable lifestyle changes, and advise seeing a doctor if risk is high or users ask clinical questions.
        Behavior:
        Keep answers concise (max 2 sentences).
        Do not give medical diagnoses or instructions.
        Highlight achievable lifestyle changes.
        Include a short motivational nudge in every answer.
        If a user asks about symptoms, medications, or is high risk, politely advise consulting a healthcare professional.
        Remind users this tool is not a substitute for a doctor.
        Tone:Friendly, encouraging, clear, concise, balanced between conversational and professional.'''
     ),
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="Answer the question based only on the following context and previous conversation.",
    ),
    ChatMessage(
        role=MessageRole.SYSTEM, content="Keep your answers short and succinct."
    ),
]

# memory
memory = ChatMemoryBuffer.from_defaults()


# bot with memory
@st.cache_resource
def init_bot():
    return ContextChatEngine(
        llm=llm, retriever=retriever, memory=memory, prefix_messages=prefix_messages
    )


rag_bot = init_bot()

##### streamlit #####

st.title("Love Your Heart: AI assistance for Heart Health")


# Display chat messages from history on app rerun
for message in rag_bot.chat_history:
    with st.chat_message(message.role):
        st.markdown(message.blocks[0].text)

# React to user input
if prompt := st.chat_input("Ask me anything!"):
    # Display user message in chat message container
    st.chat_message("human").markdown(prompt)

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Finding answers..."):
        # send question to chain to get answer
        answer = rag_bot.chat(prompt)

        # extract answer from dictionary returned by chain
        response = answer.response

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
