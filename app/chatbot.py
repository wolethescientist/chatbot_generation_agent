import os
from typing import List, Optional, Tuple
import io
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import pickle
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .website_processor import WebsiteProcessor

load_dotenv()
 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class ChatbotProcessor:
    def __init__(self):
        self.llm = GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
        # Use a smaller, faster embedding model for better performance
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        # Increase thread pool size for better parallelization
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._vector_store_cache = {}
        # Cache for text splitter to avoid recreating it
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    async def extract_text_from_pdf_binary(self, pdf_data: bytes) -> str:
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, self._extract_text_from_pdf_binary_sync, pdf_data)
        except Exception as e:
            raise ValueError(f"PDF processing error: {str(e)}")

    def _extract_text_from_pdf_binary_sync(self, pdf_data: bytes) -> str:
        pdf_stream = io.BytesIO(pdf_data)
        pdf_reader = PdfReader(pdf_stream)
        if len(pdf_reader.pages) == 0:
            raise ValueError("PDF has no pages")

        # Use list comprehension and join for better performance
        text_parts = []
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text)
                elif not page_text:
                    print(f"Warning: Page {i+1} has no extractable text")
            except Exception as e:
                print(f"Error extracting text from page {i+1}: {str(e)}")
                # Continue with other pages instead of failing completely
                continue

        if not text_parts:
            raise ValueError("No text could be extracted from the PDF")

        # Join all text parts with newlines for better structure
        return '\n'.join(text_parts)

    @lru_cache(maxsize=100)
    def create_vector_store(self, text: str) -> bytes:
        try:
            if not text or not text.strip():
                raise ValueError("Input text is empty")

            # Use the cached text splitter for better performance
            chunks = self._text_splitter.split_text(text)

            if not chunks:
                raise ValueError("Text splitting produced no chunks")

            print(f"Creating vector store with {len(chunks)} chunks...")

            # Create vector store with optimized batch processing
            vector_store = FAISS.from_texts(
                texts=chunks,
                embedding=self.embeddings
            )

            # Use more efficient serialization
            vector_store_bytes = pickle.dumps(vector_store, protocol=pickle.HIGHEST_PROTOCOL)
            if not vector_store_bytes:
                raise ValueError("Failed to serialize vector store")

            print(f"Vector store created successfully, size: {len(vector_store_bytes)} bytes")
            return vector_store_bytes
        except Exception as e:
            print(f"Vector store creation error: {str(e)}")
            raise ValueError(f"Vector store creation error: {str(e)}")
    
    def load_vector_store_from_binary(self, vector_store_data: bytes) -> FAISS:
        # Check if vector store is already in cache
        cache_key = hash(vector_store_data)
        if cache_key in self._vector_store_cache:
            return self._vector_store_cache[cache_key]
        
        # If not in cache, load and cache it
        vector_store = pickle.loads(vector_store_data)
        self._vector_store_cache[cache_key] = vector_store
        return vector_store
    
    async def process_pdf_and_create_vector_store(self, pdf_data: bytes) -> Tuple[bytes, bytes]:
        """Process PDF data and return both PDF and vector store binary data"""
        text = await self.extract_text_from_pdf_binary(pdf_data)
        loop = asyncio.get_event_loop()
        vector_store_data = await loop.run_in_executor(self._executor, self.create_vector_store, text)
        return pdf_data, vector_store_data

    async def process_website_content(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Process website URL and extract content.
        Returns (extracted_content, error_message)
        """
        website_processor = WebsiteProcessor()
        return await website_processor.process_website_url(url)

    async def process_combined_content_and_create_vector_store(
        self,
        pdf_data: Optional[bytes] = None,
        website_url: Optional[str] = None
    ) -> Tuple[bytes, bytes, Optional[str]]:
        """
        Process both PDF and website content, combine them, and create vector store.
        Returns (pdf_data, vector_store_data, website_error)
        """
        combined_text_parts = []
        website_error = None

        # Process PDF if provided
        if pdf_data:
            try:
                pdf_text = await self.extract_text_from_pdf_binary(pdf_data)
                combined_text_parts.append(f"=== PDF DOCUMENT CONTENT ===\n{pdf_text}")
                print(f"PDF content extracted: {len(pdf_text)} characters")
            except Exception as e:
                print(f"Error processing PDF: {str(e)}")
                raise ValueError(f"Error processing PDF: {str(e)}")

        # Process website if provided
        if website_url and website_url.strip():
            try:
                website_content, error = await self.process_website_content(website_url.strip())
                if website_content:
                    combined_text_parts.append(f"=== WEBSITE CONTENT ({website_url}) ===\n{website_content}")
                    print(f"Website content extracted: {len(website_content)} characters")
                else:
                    website_error = error or "Failed to extract website content"
                    print(f"Website processing failed: {website_error}")
            except Exception as e:
                website_error = f"Error processing website: {str(e)}"
                print(f"Website processing error: {website_error}")

        # Combine all content
        if not combined_text_parts:
            raise ValueError("No content available for processing (both PDF and website failed)")

        combined_text = "\n\n".join(combined_text_parts)
        print(f"Combined content length: {len(combined_text)} characters")

        # Create vector store from combined content
        loop = asyncio.get_event_loop()
        vector_store_data = await loop.run_in_executor(self._executor, self.create_vector_store, combined_text)

        return pdf_data or b'', vector_store_data, website_error
    
    async def get_chatbot_response(self, vector_store_data: bytes, query: str, chat_history: Optional[List] = None) -> str:
        if chat_history is None:
            chat_history = []

        loop = asyncio.get_event_loop()
        vector_store = await loop.run_in_executor(self._executor, self.load_vector_store_from_binary, vector_store_data)

        # The custom prompt encourages the LLM to be friendly and use its general knowledge
        # for related questions, while still prioritizing the provided context.
        prompt_template = """
        You are a friendly and helpful assistant.
        Please be conversational and greet the user if they greet you.
        
        Use the following context to answer the question. If the question is not directly answered
        by the context, use your general knowledge to provide a helpful response, but indicate that
        the information is not from the provided documents.
        
        If you don't know the answer, simply say that you don't know.
        
        Context: {context}
        Question: {question}
        Answer:
        """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
            verbose=True,  # Enable verbose output for debugging
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

        result = await loop.run_in_executor(self._executor,
            lambda: qa_chain.invoke({"question": query, "chat_history": chat_history}))

        print(f"LangChain result type: {type(result)}")
        print(f"LangChain result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        print(f"LangChain result: {result}")

        # Handle different possible response structures
        if isinstance(result, dict):
            # Try different possible keys for the answer
            answer = result.get("answer")
            if answer:
                response = str(answer)
            else:
                print(f"Warning: No 'answer' key in result: {result}")
                response = str(result)
        else:
            # If result is not a dict, convert to string
            response = str(result)
        
        # Post-process the response to make it more concise and clean
        response = self._clean_response(response)
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the response to be concise and remove unwanted phrases."""
        import re
        
        # Remove common introductory phrases
        unwanted_phrases = [
            r'^here are the[^:]*:?\s*',
            r'^here is the[^:]*:?\s*', 
            r'^the answer is[^:]*:?\s*',
            r'^based on the (context|document|information)[^:]*:?\s*',
            r'^according to the (context|document|information)[^:]*:?\s*',
            r'^from the (context|document|information)[^:]*:?\s*',
            r'^the (context|document|information) shows[^:]*:?\s*',
            r'^as mentioned in the (context|document)[^:]*:?\s*'
        ]
        
        cleaned_response = response.strip()
        
        # Remove unwanted introductory phrases (case insensitive)
        for phrase_pattern in unwanted_phrases:
            cleaned_response = re.sub(phrase_pattern, '', cleaned_response, flags=re.IGNORECASE)
        
        # Convert bullet points to numbered lists
        lines = cleaned_response.split('\n')
        numbered_lines = []
        bullet_pattern = r'^\s*[-*]\s+'
        number = 1
        
        for line in lines:
            if re.match(bullet_pattern, line):
                # Replace bullet with number
                numbered_line = re.sub(bullet_pattern, f'{number}. ', line)
                numbered_lines.append(numbered_line)
                number += 1
            else:
                numbered_lines.append(line)
                # Reset numbering if we encounter a blank line or non-bullet line
                if not line.strip():
                    number = 1
        
        cleaned_response = '\n'.join(numbered_lines)
        
        # Remove excessive special characters and clean up formatting
        # Remove multiple consecutive special characters but keep single ones that are meaningful
        cleaned_response = re.sub(r'[*]{2,}', '', cleaned_response)  # Remove multiple asterisks
        cleaned_response = re.sub(r'[#]{2,}', '', cleaned_response)  # Remove multiple hashes
        cleaned_response = re.sub(r'[-]{3,}', '', cleaned_response)  # Remove multiple dashes
        cleaned_response = re.sub(r'[=]{3,}', '', cleaned_response)  # Remove multiple equals
        cleaned_response = re.sub(r'[_]{3,}', '', cleaned_response)  # Remove multiple underscores
        
        # Clean up extra whitespace
        cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)  # Max 2 consecutive newlines
        cleaned_response = re.sub(r'\s+', ' ', cleaned_response)  # Multiple spaces to single space
        
        # Ensure the response starts with a capital letter
        cleaned_response = cleaned_response.strip()
        if cleaned_response and cleaned_response[0].islower():
            cleaned_response = cleaned_response[0].upper() + cleaned_response[1:]
        
        # Limit response length for conciseness (optional - adjust as needed)
        max_length = 500  # Adjust this value as needed
        if len(cleaned_response) > max_length:
            # Find the last complete sentence within the limit
            truncated = cleaned_response[:max_length]
            last_period = truncated.rfind('.')
            if last_period > max_length * 0.7:  # Only truncate if we don't lose too much content
                cleaned_response = truncated[:last_period + 1]
        
        return cleaned_response