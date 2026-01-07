import os
import sys

# --- HUGGING FACE CHROMADB FIX ---
# This is critical for deployment on HF Spaces to prevent SQLite errors
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ---------------------------------

import gradio as gr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Global variables to store state (sufficient for demo purposes)
vectorstore = None
qa_chain = None

def process_pdf(file_path, api_key):
    global vectorstore, qa_chain
    
    if not api_key:
        return "⚠️ Error: Please enter your OpenAI API Key first."
    
    os.environ["OPENAI_API_KEY"] = api_key
    
    try:
        # 1. Load PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # 2. Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # 3. Embed & Store (Chroma)
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        # 4. Create Retrieval Chain with "Auditor" Persona
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        
        # Custom Prompt to enforce "Audit" style behavior
        audit_template = """You are Veritas, an AI Compliance Auditor. 
        Use the following pieces of context to answer the question at the end. 
        
        RULES:
        1. If the answer is in the text, state it clearly and reference the context.
        2. If the answer is NOT in the text, you must explicitly state: "FAIL: Information not found in source document."
        3. Do not hallucinate or guess. 
        
        Context: {context}
        
        Question: {question}
        
        Verdict:"""
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(audit_template)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        return "✅ Document Processed Successfully. The Veritas Auditor is ready."
        
    except Exception as e:
        return f"❌ Error processing document: {str(e)}"

def audit_query(query):
    global qa_chain
    if not qa_chain:
        return "⚠️ Please upload a document first."
    
    try:
        response = qa_chain.invoke(query)
        return response['result']
    except Exception as e:
        return f"Error: {str(e)}"

# --- GRADIO INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="slate")) as demo:
    
    gr.Markdown(
        """
        # 🛡️ Veritas: AI Compliance Auditor
        ### Automated RAG Hallucination Detection for Financial Documentation
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            api_input = gr.Textbox(
                label="OpenAI API Key", 
                type="password", 
                placeholder="sk-..."
            )
            file_input = gr.File(
                label="Upload Financial Report (PDF)", 
                file_types=[".pdf"]
            )
            upload_btn = gr.Button("Initialize Auditor", variant="primary")
            status_output = gr.Textbox(label="System Status", interactive=False)
            
        with gr.Column(scale=2):
            query_input = gr.Textbox(label="Audit Query")
            audit_btn = gr.Button("Run Audit Check")
            response_output = gr.Textbox(label="Auditor Verdict", lines=10)

    # Button Actions
    upload_btn.click(
        process_pdf, 
        inputs=[file_input, api_input], 
        outputs=status_output
    )
    
    audit_btn.click(
        audit_query, 
        inputs=query_input, 
        outputs=response_output
    )

if __name__ == "__main__":
    demo.launch()