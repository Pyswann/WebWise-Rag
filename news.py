import bs4
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize NVIDIA NIM client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",  
    api_key=os.getenv("NVIDIA_API_KEY")
)

def call_nvidia_nim_model(input_data):
    """
    Accepts either a string (direct prompt) or a dictionary with a 'text' key.
    """
    if isinstance(input_data, dict):
        prompt = input_data.get("text", str(input_data))
    else:
        prompt = str(input_data)

    completion = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-nano-4b-v1.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4096,
        stream=False
    )
    return completion.choices[0].message.content


# Step 1: Scrape the webpage
def scrape_webpage(url, content_class="story-content no-key-elements"):
    bs4_strainer = bs4.SoupStrainer(class_=content_class)

    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs4_strainer},
        header_template={"User-Agent": "RAG-App/1.0"}
    )
    docs = loader.load()

    if not docs:
        raise ValueError(f"No content found on {url} using class '{content_class}'. Check selector or URL.")

    print("Scraped content sample:\n", docs[0].page_content[:300])
    return docs


# Step 2: Process and store data in vector store
def process_documents(docs):
    if not docs:
        raise ValueError("No documents provided for processing.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Delete existing collection before creating new one
    Chroma(
        collection_name="web_content",
        embedding_function=embeddings
    ).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="web_content"
    )
    return vectorstore


# Step 3: Set up RAG pipeline for real-time querying
def setup_rag_pipeline(vectorstore):
    llm = RunnableLambda(call_nvidia_nim_model)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    retriever = vectorstore.as_retriever()

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# Step 4: Main execution
def main(urls):
    all_docs = []

    print("\n🌐 Scraping multiple webpages...")
    for url in urls:
        print(f"📄 Scraping: {url}")
        try:
            docs = scrape_webpage(url)
            all_docs.extend(docs)
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")

    if not all_docs:
        print("⚠️ No documents were scraped.")
        return None

    print(f"\n✅ Successfully scraped {len(all_docs)} document(s).")

    print("🧠 Processing and storing data...")
    try:
        vectorstore = process_documents(all_docs)
    except Exception as e:
        print(f"Error during document processing: {e}")
        return None

    print("🧠 Vector store created successfully.\n")
    return vectorstore  # ✅ Return vectorstore, NOT the chain


# Helper function to collect URLs from user input
def get_urls_from_user():
    urls = []
    print("🔗 Enter URLs (one per line). Type 'done' when finished:")
    while True:
        user_input = input("URL (or 'done'): ").strip()
        if user_input.lower() == "done":
            break
        elif user_input.startswith("http"):
            urls.append(user_input)
        else:
            print("⚠️ Please enter a valid URL starting with http:// or https://")    
    return urls


# Run this part only when running directly
if __name__ == "__main__":
    urls = get_urls_from_user()

    if urls:
        vectorstore = main(urls)
        if vectorstore:
            rag_chain = setup_rag_pipeline(vectorstore)
        else:
            rag_chain = None
    else:
        print("❌ No URLs provided.")
        rag_chain = None

    questions = ["Give me the summary"]

    if rag_chain:
        print("\n💬 Asking questions...\n")
        for q in questions:
            print("❓ Question:", q)
            ans = rag_chain.invoke(q)
            print("✅ Answer:", ans)
            print("-" * 80)