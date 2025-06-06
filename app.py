import streamlit as st
import importlib.util
import sys

# Load news module dynamically
def load_news_module():
    module_name = "news"
    module_path = "news.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError("Could not find module 'news.py'")
    news = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = news
    spec.loader.exec_module(news)
    return news

news = load_news_module()
scrape_webpage = news.scrape_webpage
process_documents = news.process_documents
setup_rag_pipeline = news.setup_rag_pipeline


# Initialize session state
if "url_classes" not in st.session_state:
    st.session_state.url_classes = [{"url": "", "class_name": ""}]
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


# --- UI Layout ---
st.set_page_config(page_title="Dynamic RAG Q&A", layout="wide")
st.title("WebWise Rag!!")
st.text("Ft. by: Lutfor Rahman Sohan")

col_left, col_right = st.columns([1, 1.5])

with col_left:
    st.header("🔗 Enter URLs and CSS Classes")

    url_class_pairs = st.session_state.url_classes

    for i, pair in enumerate(url_class_pairs):
        st.markdown(f"### Entry {i + 1}")
        cols = st.columns([3, 2])
        url_class_pairs[i]["url"] = cols[0].text_input(f"URL #{i + 1}", value=pair["url"], key=f"url_{i}")
        url_class_pairs[i]["class_name"] = cols[1].text_input(f"Content Class #{i + 1}", value=pair["class_name"], key=f"class_{i}")

    if st.button("➕ Add Another URL + Class"):
        url_class_pairs.append({"url": "", "class_name": ""})

    if st.button("🧠 Start Processing"):
        all_docs = []
        error = False
        with st.spinner("🌐 Scraping pages..."):
            for i, pair in enumerate(url_class_pairs):
                url = pair["url"].strip()
                class_name = pair["class_name"].strip()
                if url and class_name:
                    try:
                        docs = scrape_webpage(url, class_name)
                        all_docs.extend(docs)
                    except Exception as e:
                        st.error(f"❌ Error scraping {url}: {e}")
                        error = True
                else:
                    st.warning("⚠️ Please fill both URL and Class Name for each entry.")
                    error = True
                    break

            if not error and all_docs:
                vectorstore = process_documents(all_docs)
                rag_chain = setup_rag_pipeline(vectorstore)
                st.session_state.rag_chain = rag_chain
                st.success("✅ RAG pipeline is ready!")

with col_right:
    st.header("❓ Ask a Question")

    if st.session_state.rag_chain:
        question = st.text_area("Type your question:", height=100)
        if st.button("🚀 Get Answer"):
            with st.spinner("🧠 Thinking..."):
                try:
                    answer = st.session_state.rag_chain.invoke(question)
                    st.markdown("### ✅ Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"❌ Error getting answer: {e}")
    else:
        st.info("ℹ️ Finish processing URLs to ask questions.")

# Store updated URL-class pairs
st.session_state.url_classes = url_class_pairs