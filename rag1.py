from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Set user agent
os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
)

#  Load and split website content
print(" Loading content from website...")
loader = WebBaseLoader(web_paths=["https://www.educosys.com/course/genai"])
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
print(f" Loaded {len(all_splits)} document chunks.")

#  Embeddings & vector store
print(" Embedding and storing chunks...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = Chroma(
    collection_name="educosys_genai_info",
    embedding_function=embeddings,
    persist_directory="./chroma_genai"
)

vectorstore.add_documents(documents=all_splits)
print(f" Total stored chunks: {vectorstore._collection.count()}")


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#  Tool function (uses preloaded retriever)
@tool
def retrieve_context(query: str):
    """Retrieve and summarize Gen AI course details from the website."""
    try:
        print(f"\n🔍 Querying: {query}")
        results = retriever.invoke(query)
        print(f" Retrieved {len(results)} matching document(s).")

        if not results:
            return f" No information found for: '{query}'."

        structured_response = "###  Retrieved Information:\n"
        for idx, doc in enumerate(results, 1):
            snippet = doc.page_content.strip().replace("\n", " ")
            structured_response += f"\n**Result {idx}:**\n{snippet[:500]}...\n"

        return structured_response

    except Exception as e:
        return f" Error: {e}"


# Step 5: Create agent
print(" Initializing agent...")
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
agent_executor = create_react_agent(llm, [retrieve_context])

# Step 6: Run user query
input_message = "give the topics that will be covered in week 6"

print("\n Running agent...\n")
for event in agent_executor.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values"
):
    event["messages"][-1].pretty_print()
