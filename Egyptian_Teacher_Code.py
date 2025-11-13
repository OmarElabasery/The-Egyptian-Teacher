import os
import logging
import warnings
import gdown
import fitz  # PyMuPDF
import arabic_reshaper
from bidi.algorithm import get_display
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
import streamlit as st
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

class EgyptianTeacherRAG:
    def __init__(self, azure_endpoint: str, api_key: str, api_version: str, output_folder: str = "egyptian_courses",
                 db_directory: str = "db", cache_folder: str = "model_cache"):
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.output_folder = Path(output_folder)
        self.db_directory = Path(db_directory)
        self.cache_folder = Path(cache_folder)
        self.chats_folder = Path("saved_chats")
        self.chats_folder.mkdir(exist_ok=True)

        # Create necessary directories
        for directory in [self.output_folder, self.db_directory, self.cache_folder]:
            directory.mkdir(exist_ok=True)

        # Set HuggingFace cache directory
        os.environ["TRANSFORMERS_CACHE"] = str(self.cache_folder)

    
    def download_materials(self, folder_url: str) -> bool:
        """
        Download materials from Google Drive folder.
        """
        try:
            logger.info(f"Downloading materials from {folder_url}")
            gdown.download_folder(folder_url, output=str(self.output_folder), quiet=False)
            logger.info(f"Materials downloaded to {self.output_folder}")
            return True
        except Exception as e:
            logger.error(f"Error downloading materials: {str(e)}")
            return False

    def extract_text_from_pdfs(self) -> Optional[str]:
        """
        Extract and process text from PDF files with Arabic support.
        """
        try:
            all_text = []
            pdf_files = list(self.output_folder.glob("*.pdf"))

            if not pdf_files:
                logger.warning("No PDF files found in the specified folder")
                return None

            for pdf_file in pdf_files:
                logger.info(f"Processing PDF: {pdf_file.name}")
                doc = fitz.Document(pdf_file)  # Updated to use fitz.Document

                for page_num, page in enumerate(doc, 1):
                    try:
                        raw_text = page.get_text()
                        if raw_text.strip():  # Check if text is not empty
                            reshaped_text = arabic_reshaper.reshape(raw_text)
                            bidi_text = get_display(reshaped_text)
                            all_text.append(bidi_text)
                    except Exception as e:
                        logger.error(f"Error processing page {page_num} in {pdf_file.name}: {str(e)}")

                doc.close()

            return "\n".join(all_text)
        except Exception as e:
            logger.error(f"Error in text extraction: {str(e)}")
            return None


    def split_text(self, materials_text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """
        Split text into manageable chunks.
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " "]
            )
            chunks = splitter.split_text(materials_text)
            logger.info(f"Split materials into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            return []
        
    def load_vectorstore(self) -> Optional[Chroma]:
        try:
            # Create embeddings explicitly
            embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large",
                model_kwargs={'device': 'cpu'},
                cache_folder=str(self.cache_folder),
                encode_kwargs={'normalize_embeddings': True}
            )

            # Check if the vector store directory exists and contains necessary files
            if (self.db_directory / "chroma.sqlite3").exists():
                logger.info("Loading existing vector store...")
                vectorstore = Chroma(
                    persist_directory=str(self.db_directory),
                    embedding_function=embeddings
                )
                return vectorstore
            else:
                logger.warning("Chroma vector store files not found. Please create the vector store first.")
                return None

        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None

    def setup_embeddings_and_vectorstore(self, chunks: List[str]) -> Tuple[Optional[Chroma], Optional[HuggingFaceEmbeddings]]:
        """
        Set up HuggingFace embeddings and vector store.
        """
        try:
            # Check if vector store already exists
            if (self.db_directory / "chroma.sqlite3").exists():
                logger.info("Loading existing vector store...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="intfloat/multilingual-e5-large",
                    model_kwargs={'device': 'cpu'},
                    cache_folder=str(self.cache_folder),
                    encode_kwargs={'normalize_embeddings': True}
                )
                vectorstore = Chroma(
                    persist_directory=str(self.db_directory),
                    embedding_function=embeddings
                )
                return vectorstore, embeddings

            # If no existing vector store, create a new one
            embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large",
                model_kwargs={'device': 'cpu'},
                cache_folder=str(self.cache_folder),
                encode_kwargs={'normalize_embeddings': True}
            )

            vectorstore = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings,
                persist_directory=str(self.db_directory)
            )
            vectorstore.persist()
            logger.info("Vector store created and persisted successfully")
            return vectorstore, embeddings
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            return None, None
    def build_qa_chain(self, vectorstore: Chroma, persona: str) -> Optional[RetrievalQA]:
        try:
            llm = AzureChatOpenAI(
                model_name="gpt-35-turbo-16k",
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
                temperature=0.2,  # Slight increase in creativity
            )

            prompts = {
                "funny": PromptTemplate(
                    template="""انت مدرس مصري دمك خفيف وبتتكلم بالعامية.
                    استخدم المعلومات التالية لشرح الدرس بشكل بسيط ومضحك، ولكن بدون مبالغة:

                    تاريخ المحادثة السابقة:
                    {chat_history}

                    السياق: {context}

                    السؤال: {question}

                    الإجابة:""",
                    input_variables=["chat_history", "context", "question"],
                ),
                "serious": PromptTemplate(
                    template="""انت مدرس مصري جاد وبتتكلم بالعامية.
                    استخدم المعلومات التالية للرد بشكل جاد:

                    تاريخ المحادثة السابقة:
                    {chat_history}

                    السياق: {context}

                    السؤال: {question}

                    الإجابة:""",
                    input_variables=["chat_history", "context", "question"],
                ),
                "friendly": PromptTemplate(
                    template="""انت مدرس مصري زي صحابك وبتتكلم بالعامية.
                    استخدم المعلومات التالية للرد بطريقة ودية ومريحة:

                    تاريخ المحادثة السابقة:
                    {chat_history}

                    السياق: {context}

                    السؤال: {question}

                    الإجابة:""",
                    input_variables=["chat_history", "context", "question"],
                ),
            }

            def create_retrieval_qa_with_history(llm, retriever, persona):
                def retrieve_and_answer(inputs):
                    query = inputs['query']
                    chat_history = inputs.get('chat_history', '')
                    
                    # Retrieve relevant documents
                    docs = retriever.get_relevant_documents(query)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Prepare the full prompt
                    full_prompt = prompts[persona].format(
                        chat_history=chat_history,
                        context=context,
                        question=query
                    )
                    
                    # Generate the response
                    response = llm.invoke(full_prompt).content
                    return response
                
                return retrieve_and_answer

            # Create a custom retrieval QA function
            qa_chain = create_retrieval_qa_with_history(
                llm=llm, 
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), 
                persona=persona
            )
            
            logger.info(f"QA chain built for persona: {persona}")
            return qa_chain
        except Exception as e:
            logger.error(f"Error building QA chain: {str(e)}")
            return None
    def initialize_system(self, folder_url: str) -> Optional[Chroma]:
        """
        Initialize the complete RAG system.
        """
        # Download materials if they don't exist
        if not any(self.output_folder.glob("*.pdf")):
            if not self.download_materials(folder_url):
                return None

        # Extract text
        materials_text = self.extract_text_from_pdfs()
        if not materials_text:
            return None

        # Split text
        chunks = self.split_text(materials_text)
        if not chunks:
            return None

        # Setup vector store
        vectorstore, _ = self.setup_embeddings_and_vectorstore(chunks)
        return vectorstore
    
    def get_welcome_message(self, persona: str) -> str:
        """
        Get the appropriate welcome message based on the selected persona.
        """
        welcome_messages = {
            "funny": " أهلا بيك يا صديقي! أنا مدرسك اللي دمه خفيف وجاهز أساعدك في المذاكرة بطريقة حلوة ومرحة. يلا نبدأ!",
            "serious": "📚 السلام عليكم. أنا مدرسك وجاهز لمساعدتك في فهم المادة العلمية بشكل جاد ودقيق. تفضل باسألتك.",
            "friendly": "👋 أهلا يا صديقي! أنا مدرسك وصديقك، وجاهز نذاكر مع بعض ونفهم المادة بطريقة سهلة وبسيطة. ابدأ باللي عايز تفهمه!"
        }
        return welcome_messages.get(persona, welcome_messages["friendly"])

    def format_chat_name(self, filename: str) -> str:
        """
        Format the chat filename for display.
        """
        try:
            # Remove timestamp and .json extension
            name = filename.rsplit('_', 1)[0].replace('.json', '')
            # Convert underscores to spaces
            return name.replace('_', ' ')
        except:
            return filename

    def save_chat(self, chat_history: List[Dict], chat_name: str) -> str:
        """
        Save chat history to a JSON file with a custom name.
        """
        try:
            # Clean the filename
            clean_name = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in chat_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{clean_name}_{timestamp}.json"
            
            filepath = self.chats_folder / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'name': chat_name,
                    'timestamp': timestamp,
                    'history': chat_history
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Chat saved to {filepath}")
            return filename
        except Exception as e:
            logger.error(f"Error saving chat: {str(e)}")
            return None

    def load_chat(self, filename: str) -> List[Dict]:
        """
        Load chat history from a JSON file.
        """
        try:
            filepath = self.chats_folder / filename
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data['history']
        except Exception as e:
            logger.error(f"Error loading chat: {str(e)}")
            return []

    def delete_chat(self, filename: str) -> bool:
        """
        Delete a saved chat file.
        """
        try:
            filepath = self.chats_folder / filename
            filepath.unlink()
            logger.info(f"Deleted chat file: {filename}")
            return True
        except Exception as e:
            logger.error(f"Error deleting chat: {str(e)}")
            return False

    def list_saved_chats(self) -> List[str]:
        """
        List all saved chat files.
        """
        try:
            return [f.name for f in self.chats_folder.glob("*.json")]
        except Exception as e:
            logger.error(f"Error listing chats: {str(e)}")
            return []

# Streamlit App
def main():
    # Set page configuration
    st.set_page_config(page_title="المدرس المصري", page_icon="📚", layout="wide")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = None
    if 'welcome_shown' not in st.session_state:
        st.session_state.welcome_shown = False
    if 'new_chat_name' not in st.session_state:
        st.session_state.new_chat_name = ""

    st.title("📚 المدرس المصري - Egyptian Teacher RAG App")

    # Initialize RAG system
    AZURE_ENDPOINT = "https://mopchatbot.openai.azure.com/"
    API_KEY = "a920046601144cfeb17de716f9dc8610"
    API_VERSION = "2024-02-15-preview"
    FOLDER_URL = "https://drive.google.com/drive/u/1/folders/1M4fjc-cBNLtQ-Msz290HFaY5au_Szz2y"

    PERSONA_CHOICES = {"مدرس دم خفيف": "funny", "مدرس جاد": "serious", "مدرس زي صحابك": "friendly"}
    rag_system = EgyptianTeacherRAG(azure_endpoint=AZURE_ENDPOINT, api_key=API_KEY, api_version=API_VERSION)

    # Sidebar organization
    with st.sidebar:
        st.title("إعدادات المحادثة")
        
        # 1. Teacher Persona Selection
        st.header("👨‍🏫 شخصية المدرس")
        persona = st.selectbox(
            "اختر شخصية المدرس المفضلة لديك",
            list(PERSONA_CHOICES.keys()),
            key="teacher_persona"
        )
        persona_key = PERSONA_CHOICES[persona]
        
        st.markdown("---")

         # 2. Chat Management
        st.header("💬 المحادثات")
        
        # New chat button
        if st.button("📝 محادثة جديدة", key="new_chat"):
            st.session_state.chat_history = []
            st.session_state.current_chat = None
            st.session_state.welcome_shown = False
            st.rerun()

        # Save current chat
        if st.session_state.chat_history:
            st.text_input("اسم المحادثة", key="new_chat_name")
            if st.button("💾 حفظ المحادثة الحالية"):
                if st.session_state.new_chat_name:
                    filename = rag_system.save_chat(st.session_state.chat_history, st.session_state.new_chat_name)
                    if filename:
                        st.success("تم حفظ المحادثة بنجاح!")
                else:
                    st.warning("برجاء إدخال اسم للمحادثة")

        # Saved Chats Section
        st.subheader("المحادثات المحفوظة")
        saved_chats = rag_system.list_saved_chats()
        
        if saved_chats:
            for chat_file in saved_chats:
                chat_name = rag_system.format_chat_name(chat_file)
                if st.button(f"📜 {chat_name}", key=f"chat_{chat_file}"):
                    st.session_state.chat_history = rag_system.load_chat(chat_file)
                    st.session_state.current_chat = chat_file
                    st.session_state.welcome_shown = True
                    st.rerun()

        else:
            st.info("لا توجد محادثات محفوظة")

    vectorstore = rag_system.load_vectorstore()
    if not vectorstore:
        vectorstore = rag_system.initialize_system(FOLDER_URL)
    
    if vectorstore:
        qa_chain = rag_system.build_qa_chain(vectorstore, persona_key)

        if qa_chain:
            # Show welcome message for new chats
            if not st.session_state.welcome_shown and not st.session_state.chat_history:
                welcome_msg = rag_system.get_welcome_message(persona_key)
                st.session_state.chat_history.append({
                    "role": "AI",
                    "content": welcome_msg
                })
                st.session_state.welcome_shown = True

            # Display chat history
            for msg in st.session_state.chat_history:
                with st.chat_message(msg['role']):
                    st.markdown(msg['content'])

            # Chat input
            if user_query := st.chat_input("اسأل المدرس المصري"):
                with st.chat_message("Human"):
                    st.markdown(user_query)
                
                st.session_state.chat_history.append({
                    "role": "Human", 
                    "content": user_query
                })

                try:
                    chat_history_str = "\n".join([
                        f"{'المستخدم' if msg['role'] == 'Human' else 'المدرس'}: {msg['content']}" 
                        for msg in st.session_state.chat_history[:-1]
                    ])

                    with st.chat_message("AI"):
                        with st.spinner("جاري التفكير..."):
                            result = qa_chain({
                                'query': user_query, 
                                'chat_history': chat_history_str
                            })
                        st.markdown(result)

                    st.session_state.chat_history.append({
                        "role": "AI", 
                        "content": result
                    })

                except Exception as e:
                    st.error(f"حدث خطأ أثناء معالجة السؤال: {str(e)}")
    else:
        st.warning("لم يتم العثور على بيانات. برجاء التأكد من وجود ملفات التدريب.")

if __name__ == "__main__":
    main()