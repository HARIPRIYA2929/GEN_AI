from dotenv import load_dotenv
from yarl import Query
load_dotenv()

import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
from langchain.chat_models import init_chat_model

import base64

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_chroma import Chroma




loader = WebBaseLoader(
   web_paths=["https://www.educosys.com/course/genai"]
)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)


print(all_splits)


from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


vectorstore = Chroma(collection_name="educosys_genai_info", embedding_function=embeddings, persist_directory="./chroma_genai")


vectorstore.add_documents(documents=all_splits)


print(vectorstore._collection.count())  
# Check total stored chunks


@tool
def retrieve_context(query: str):
   """ Give me the duration of the gen ai course """
   try:
       embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-3-large")
       vector_store = Chroma(
           collection_name="educosys_genai_info",
           embedding_function=embeddings,
           persist_directory="./chroma_genai",
       )
       retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


       print(f"Querying retrieve_context with: {query}")
       print("--------------------------------------------------------------")
       results = retriever.invoke(query)
       print(f"Retrieved documents: {len(results)} matches found")
       for i, doc in enumerate(results):
           print(f"Document {i + 1}: {doc.page_content[:100]}...")
      
       print("--------------------------------------------------------------")


       content = "\n".join([doc.page_content for doc in results])
       if not content:
           print(f"No content retrieved for query: {query}")
           return f"No reviews found for '{query}'."
      
       print("--------------------------------------------------------------")
       print(f"Returning content: {content[:200]}...")
       return content
   except Exception as e:
       print(f"Error in retrieve_context: {e}")
       return f"Error retrieving reviews for '{query}'. Please try again."
   llm = init_chat_model("gemini-2.0-flash", model_provider="google")
   agent_executor = create_react_agent(llm, [retrieve_context])
   input_message = (
   "Give me the duration of the gen ai course"
)
   for event in agent_executor.stream(
   {"messages": [{"role": "user", "content": input_message}]},
   stream_mode="values"
):
       event["messages"][-1].pretty_print()
