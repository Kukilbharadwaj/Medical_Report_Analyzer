"""
Advanced RAG System with Medical Image Analysis
Supports hybrid search, cross-encoder reranking, and LLaMA Vision for medical image analysis
"""

import os
import io
import base64
from typing import List, Tuple, Optional
from dataclasses import dataclass

# PDF Loading
from langchain_community.document_loaders.pdf import PyPDFLoader

# Text Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector Store
from langchain_community.vectorstores import FAISS

# BM25 for keyword search
from rank_bm25 import BM25Okapi

# Reranking
from sentence_transformers import CrossEncoder

# Groq LLM
from groq import Groq

# Environment variables
from dotenv import load_dotenv

# Utilities
import numpy as np
from pathlib import Path

# Image processing
import fitz  # PyMuPDF for PDF image extraction
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Load environment variables
load_dotenv()


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with its score"""
    content: str
    page_number: int
    score: float
    source: str  # 'vector', 'bm25', or 'hybrid'


@dataclass
class ExtractedImage:
    """Represents an extracted image from PDF"""
    image: Image.Image
    page_number: int
    image_index: int
    embedding: Optional[np.ndarray] = None
    base64_data: Optional[str] = None
    description: Optional[str] = None


class AdvancedRAGSystem:
    """
    Advanced RAG System with:
    - Hybrid Search (Vector + BM25)
    - Cross-Encoder Reranking
    - Groq LLM for generation
    - Image extraction and analysis with CLIP + LLaMA Vision
    """
    
    def __init__(
        self,
        groq_api_key: str = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model: str = "llama-3.3-70b-versatile",
        vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        clip_model: str = "openai/clip-vit-base-patch32",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        vector_k: int = 10,
        bm25_k: int = 10,
        final_k: int = 5,
        vector_weight: float = 0.5,
        enable_image_support: bool = True
    ):
        """
        Initialize the Advanced RAG System
        
        Args:
            groq_api_key: Groq API key (uses env var if not provided)
            embedding_model: HuggingFace embedding model name
            reranker_model: Cross-encoder model for reranking
            llm_model: Groq model name for text generation
            vision_model: Groq vision model for image analysis
            clip_model: CLIP model for image embeddings
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            vector_k: Number of results from vector search
            bm25_k: Number of results from BM25 search
            final_k: Final number of chunks after reranking
            vector_weight: Weight for vector search in hybrid (0-1)
            enable_image_support: Whether to enable image extraction and analysis
        """
        # API Key
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required. Set it in .env file or pass it directly.")
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=self.groq_api_key)
        self.llm_model = llm_model
        self.vision_model = vision_model
        
        # Image support
        self.enable_image_support = enable_image_support
        self.extracted_images: List[ExtractedImage] = []
        self.image_embeddings = None
        
        # Initialize CLIP for image embeddings
        if self.enable_image_support:
            print(f"Loading CLIP model: {clip_model}")
            self.clip_model = CLIPModel.from_pretrained(clip_model)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
            self.clip_model.eval()
        else:
            self.clip_model = None
            self.clip_processor = None
        
        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Retrieval parameters
        self.vector_k = vector_k
        self.bm25_k = bm25_k
        self.final_k = final_k
        self.vector_weight = vector_weight
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize reranker
        print(f"Loading reranker model: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model, max_length=512)
        
        # Storage
        self.vectorstore = None
        self.bm25 = None
        self.chunks = []
        self.chunk_texts = []
        self.is_loaded = False
        
        print("RAG System initialized successfully!")
        if self.enable_image_support:
            print("Image support enabled with CLIP embeddings + LLaMA Vision")
    
    def _extract_images_from_pdf(self, pdf_path: str, min_width: int = 100, min_height: int = 100) -> List[ExtractedImage]:
        """
        Extract images from PDF using PyMuPDF.
        Uses embedded image extraction first, then falls back to page rendering
        for scanned/image-only PDFs.
        
        Args:
            pdf_path: Path to PDF file
            min_width: Minimum image width to extract
            min_height: Minimum image height to extract
            
        Returns:
            List of ExtractedImage objects
        """
        images = []
        doc = None
        
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Warning: Could not open PDF for image extraction: {e}")
            return images
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_had_images = False
                
                # Method 1: Try extracting embedded images
                try:
                    image_list = page.get_images(full=True)
                except Exception as e:
                    print(f"Warning: Could not get image list from page {page_num + 1}: {e}")
                    image_list = []
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        if not img_info or len(img_info) == 0:
                            continue
                        
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        
                        if not base_image or "image" not in base_image:
                            continue
                        
                        image_bytes = base_image["image"]
                        if not image_bytes:
                            continue
                        
                        # Convert to PIL Image
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        
                        # Filter small images (likely icons/logos)
                        if pil_image.width < min_width or pil_image.height < min_height:
                            continue
                        
                        # Convert to RGB if necessary
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        
                        # Convert to base64 for API calls
                        buffered = io.BytesIO()
                        pil_image.save(buffered, format="JPEG", quality=85)
                        base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        
                        images.append(ExtractedImage(
                            image=pil_image,
                            page_number=page_num + 1,
                            image_index=img_index,
                            base64_data=base64_data
                        ))
                        page_had_images = True
                        
                    except Exception as e:
                        print(f"Warning: Could not extract image {img_index} from page {page_num + 1}: {e}")
                        continue
                
                # Method 2: If no embedded images found AND page has little/no text,
                # render page as image (handles scanned PDFs / image-only PDFs)
                if not page_had_images:
                    try:
                        # Check if page has significant text content
                        page_text = page.get_text().strip()
                        has_text = len(page_text) > 50  # More than ~50 chars indicates real text
                        
                        # Only render page as image if it lacks text (scanned/image PDF)
                        if not has_text:
                            # Render page at 150 DPI
                            mat = fitz.Matrix(150 / 72, 150 / 72)
                            pix = page.get_pixmap(matrix=mat)
                            
                            if pix.width >= min_width and pix.height >= min_height:
                                img_bytes = pix.tobytes("jpeg")
                                pil_image = Image.open(io.BytesIO(img_bytes))
                                
                                if pil_image.mode != 'RGB':
                                    pil_image = pil_image.convert('RGB')
                                
                                base64_data = base64.b64encode(img_bytes).decode('utf-8')
                                
                                images.append(ExtractedImage(
                                    image=pil_image,
                                    page_number=page_num + 1,
                                    image_index=0,
                                    base64_data=base64_data
                                ))
                    except Exception as e:
                        print(f"Warning: Could not render page {page_num + 1} as image: {e}")
                        continue
        
        finally:
            if doc:
                doc.close()
        
        return images
    
    def _compute_clip_embeddings(self, images: List[ExtractedImage]) -> np.ndarray:
        """
        Compute CLIP embeddings for extracted images
        
        Args:
            images: List of ExtractedImage objects
            
        Returns:
            numpy array of image embeddings
        """
        if not images or not self.clip_model:
            return np.array([])
        
        embeddings = []
        
        with torch.no_grad():
            for img in images:
                try:
                    inputs = self.clip_processor(images=img.image, return_tensors="pt")
                    image_features = self.clip_model.get_image_features(**inputs)
                    # Handle case where model returns output object instead of tensor
                    if hasattr(image_features, 'pooler_output'):
                        image_features = image_features.pooler_output
                    elif hasattr(image_features, 'last_hidden_state'):
                        image_features = image_features.last_hidden_state[:, 0, :]
                    # Normalize embedding using torch.linalg.norm for compatibility
                    norm = torch.linalg.norm(image_features, dim=-1, keepdim=True)
                    image_features = image_features / norm.clamp(min=1e-8)
                    embeddings.append(image_features.cpu().numpy().flatten())
                    img.embedding = embeddings[-1]
                except Exception as e:
                    print(f"Warning: Could not compute CLIP embedding for image: {e}")
                    continue
        
        return np.array(embeddings) if embeddings else np.array([])
    
    def _get_text_clip_embedding(self, text: str) -> np.ndarray:
        """
        Compute CLIP text embedding for image search
        
        Args:
            text: Query text
            
        Returns:
            Text embedding
        """
        if not self.clip_model:
            return np.array([])
        
        with torch.no_grad():
            try:
                inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
                text_features = self.clip_model.get_text_features(**inputs)
                # Handle case where model returns output object instead of tensor
                if hasattr(text_features, 'pooler_output'):
                    text_features = text_features.pooler_output
                elif hasattr(text_features, 'last_hidden_state'):
                    text_features = text_features.last_hidden_state[:, 0, :]
                # Normalize embedding using torch.linalg.norm for compatibility
                norm = torch.linalg.norm(text_features, dim=-1, keepdim=True)
                text_features = text_features / norm.clamp(min=1e-8)
                return text_features.cpu().numpy().flatten()
            except Exception as e:
                print(f"Warning: Could not compute CLIP text embedding: {e}")
                return np.array([])
    
    def search_images(self, query: str, top_k: int = 3) -> List[Tuple[ExtractedImage, float]]:
        """
        Search for relevant images using CLIP embeddings
        
        Args:
            query: Text query
            top_k: Number of top images to return
            
        Returns:
            List of (ExtractedImage, similarity_score) tuples
        """
        if not self.extracted_images or self.image_embeddings is None:
            return []
        
        # Get text embedding
        text_embedding = self._get_text_clip_embedding(query)
        
        # Compute cosine similarities
        similarities = np.dot(self.image_embeddings, text_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum threshold
                results.append((self.extracted_images[idx], float(similarities[idx])))
        
        return results
    
    def analyze_image_with_vision(self, image: ExtractedImage, query: str = None) -> str:
        """
        Analyze a medical image using Groq's LLaMA Vision model
        
        Args:
            image: ExtractedImage object
            query: Optional specific question about the image
            
        Returns:
            Analysis text from the vision model
        """
        if not image.base64_data:
            return "Error: Image data not available"
        
        # Build the prompt
        if query:
            prompt = f"""You are a medical imaging expert. Analyze this medical image and answer the following question:

Question: {query}

Provide a detailed, professional analysis focusing on:
1. Direct answer to the question
2. Relevant observations from the image
3. Any notable findings or abnormalities"""
        else:
            prompt = """You are a medical imaging expert. Analyze this medical image and provide:

1. **Image Type**: Identify the type of medical image (X-ray, MRI, CT scan, ultrasound, etc.)
2. **Body Region**: Identify the anatomical region shown
3. **Key Observations**: Describe what you observe in the image
4. **Notable Findings**: Point out any abnormalities or areas of concern
5. **Image Quality**: Comment on the technical quality of the image

Be thorough but concise in your analysis."""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image.base64_data}"
                                }
                            }
                        ]
                    }
                ],
                model=self.vision_model,
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def analyze_all_images(self) -> List[Tuple[ExtractedImage, str]]:
        """
        Analyze all extracted images using vision model
        
        Returns:
            List of (image, analysis) tuples
        """
        results = []
        for img in self.extracted_images:
            analysis = self.analyze_image_with_vision(img)
            img.description = analysis
            results.append((img, analysis))
        return results
    
    def load_image(self, image_path: str = None, image_data: bytes = None, image_name: str = "uploaded_image") -> int:
        """
        Load a standalone image file for analysis.
        
        Args:
            image_path: Path to image file (optional if image_data provided)
            image_data: Raw image bytes (optional if image_path provided)
            image_name: Name for the image
            
        Returns:
            Number of images loaded
        """
        try:
            if image_path:
                pil_image = Image.open(image_path)
            elif image_data:
                pil_image = Image.open(io.BytesIO(image_data))
            else:
                raise ValueError("Either image_path or image_data must be provided")
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=85)
            base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            img = ExtractedImage(
                image=pil_image,
                page_number=0,  # standalone image
                image_index=len(self.extracted_images),
                base64_data=base64_data
            )
            
            self.extracted_images.append(img)
            
            # Compute CLIP embedding for the new image
            if self.clip_model:
                self._compute_clip_embeddings([img])
                # Update the combined embeddings array
                all_embeddings = [i.embedding for i in self.extracted_images if i.embedding is not None]
                if all_embeddings:
                    self.image_embeddings = np.array(all_embeddings)
            
            self.is_loaded = True
            print(f"Image loaded: {image_name}")
            return 1
            
        except Exception as e:
            print(f"Error loading image: {e}")
            raise
    
    def load_images(self, image_files: list) -> int:
        """
        Load multiple standalone images.
        
        Args:
            image_files: List of dicts with 'data' (bytes) and 'name' (str)
            
        Returns:
            Number of images loaded
        """
        count = 0
        for img_file in image_files:
            try:
                self.load_image(
                    image_data=img_file.get('data'),
                    image_name=img_file.get('name', f'image_{count}')
                )
                count += 1
            except Exception as e:
                print(f"Warning: Failed to load image {img_file.get('name', '?')}: {e}")
        return count
    
    def load_pdf(self, pdf_path: str) -> int:
        """
        Load and process a PDF document.
        Handles text-only, image-only, and mixed PDFs gracefully.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of chunks created
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        print(f"\n{'='*50}")
        print(f"Loading PDF: {pdf_path.name}")
        print(f"{'='*50}")
        
        # Load PDF text
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            print(f"Pages loaded: {len(documents)}")
        except Exception as e:
            print(f"Warning: Could not extract text from PDF: {e}")
            documents = []
        
        # Split into chunks (filter out empty/whitespace-only content)
        all_chunks = self.text_splitter.split_documents(documents) if documents else []
        self.chunks = [c for c in all_chunks if c.page_content.strip()]
        self.chunk_texts = [chunk.page_content for chunk in self.chunks]
        print(f"Text chunks created: {len(self.chunks)}")
        
        # Create vector store only if we have text chunks
        if self.chunks:
            print("Creating vector embeddings...")
            self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
            
            print("Building BM25 index...")
            tokenized_chunks = [text.lower().split() for text in self.chunk_texts]
            self.bm25 = BM25Okapi(tokenized_chunks)
        else:
            print("No text content found in PDF (image-only document).")
            self.vectorstore = None
            self.bm25 = None
        
        # Extract images if enabled
        if self.enable_image_support:
            print("Extracting images from PDF...")
            self.extracted_images = self._extract_images_from_pdf(str(pdf_path))
            print(f"Images extracted: {len(self.extracted_images)}")
            
            if self.extracted_images:
                print("Computing CLIP embeddings for images...")
                self.image_embeddings = self._compute_clip_embeddings(self.extracted_images)
                print(f"Image embeddings computed: {len(self.image_embeddings)}")
        
        # Mark as loaded if we have either text or images
        if self.chunks or self.extracted_images:
            self.is_loaded = True
            print(f"\nDocument processed successfully!")
            print(f"Ready to answer questions about: {pdf_path.name}")
        else:
            raise ValueError("PDF contains no extractable text or images.")
        
        return len(self.chunks)
    
    def _vector_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Perform vector similarity search"""
        if not self.vectorstore or not self.chunk_texts:
            return []
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        # Find indices and normalize scores
        indices_scores = []
        for doc, score in results:
            try:
                idx = self.chunk_texts.index(doc.page_content)
                # FAISS returns L2 distance, convert to similarity
                similarity = 1 / (1 + score)
                indices_scores.append((idx, similarity))
            except ValueError:
                continue
        return indices_scores
    
    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Perform BM25 keyword search"""
        if not self.bm25 or not self.chunk_texts:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        if len(scores) == 0:
            return []
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Normalize scores
        max_score = max(scores) if max(scores) > 0 else 1
        indices_scores = [(int(idx), scores[idx] / max_score) for idx in top_indices if scores[idx] > 0]
        
        return indices_scores
    
    def _hybrid_search(self, query: str) -> List[Tuple[int, float]]:
        """
        Combine vector and BM25 search with Reciprocal Rank Fusion (RRF)
        """
        vector_results = self._vector_search(query, self.vector_k)
        bm25_results = self._bm25_search(query, self.bm25_k)
        
        # Reciprocal Rank Fusion
        k_rrf = 60  # RRF constant
        scores = {}
        
        # Add vector search scores
        for rank, (idx, _) in enumerate(vector_results):
            scores[idx] = scores.get(idx, 0) + self.vector_weight / (k_rrf + rank + 1)
        
        # Add BM25 scores
        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + (1 - self.vector_weight) / (k_rrf + rank + 1)
        
        # Sort by combined score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def _rerank(self, query: str, candidates: List[Tuple[int, float]]) -> List[RetrievedChunk]:
        """
        Rerank candidates using cross-encoder
        """
        if not candidates:
            return []
        
        # Prepare pairs for reranking
        pairs = [(query, self.chunk_texts[idx]) for idx, _ in candidates]
        
        # Get reranker scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine with original indices
        reranked = []
        for i, (idx, _) in enumerate(candidates):
            chunk = self.chunks[idx]
            page_num = chunk.metadata.get('page', 0) + 1
            reranked.append(RetrievedChunk(
                content=self.chunk_texts[idx],
                page_number=page_num,
                score=float(rerank_scores[i]),
                source='hybrid'
            ))
        
        # Sort by reranker score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        return reranked[:self.final_k]
    
    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """
        Full retrieval pipeline: Hybrid Search -> Reranking
        
        Args:
            query: User query
            
        Returns:
            List of retrieved and reranked chunks
        """
        if not self.is_loaded:
            raise RuntimeError("No document loaded. Call load_pdf() first.")
        
        # Hybrid search
        candidates = self._hybrid_search(query)
        
        # Rerank
        reranked_chunks = self._rerank(query, candidates)
        
        return reranked_chunks
    
    def _build_prompt(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Build the prompt for the LLM"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Source {i} - Page {chunk.page_number}]\n{chunk.content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context.

INSTRUCTIONS:
1. Use ONLY the information from the context below to answer
2. If the answer is not in the context, say "I cannot find this information in the document"
3. Be specific and cite the page numbers when possible
4. Provide a comprehensive but concise answer

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    def generate_answer(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """
        Generate answer using Groq LLM
        
        Args:
            query: User question
            chunks: Retrieved context chunks
            
        Returns:
            Generated answer
        """
        prompt = self._build_prompt(query, chunks)
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided document context. Be accurate and cite sources when possible."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.llm_model,
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(self, question: str, show_sources: bool = True) -> str:
        """
        Full RAG pipeline: Retrieve -> Rerank -> Generate
        Automatically falls back to image-only analysis if no text is available.
        
        Args:
            question: User question
            show_sources: Whether to show source information
            
        Returns:
            Generated answer with optional sources
        """
        if not self.is_loaded:
            return "Please load a PDF document or images first."
        
        # If we have text chunks, use text retrieval
        chunks = []
        if self.vectorstore and self.chunk_texts:
            chunks = self.retrieve(question)
        
        # If no text chunks but we have images, auto-redirect to image analysis
        if not chunks and self.extracted_images and self.enable_image_support:
            return self._image_only_query(question, show_sources)
        
        if not chunks:
            return "No relevant information found in the document."
        
        # Generate answer
        answer = self.generate_answer(question, chunks)
        
        # Add sources if requested
        if show_sources:
            sources = "\n\n📚 **Sources:**\n"
            for i, chunk in enumerate(chunks, 1):
                sources += f"  [{i}] Page {chunk.page_number} (relevance: {chunk.score:.3f})\n"
            answer += sources
        
        return answer
    
    def _image_only_query(self, question: str, show_sources: bool = True) -> str:
        """
        Handle queries when only images are available (no text content).
        Searches for relevant images and analyzes them with vision model.
        
        Args:
            question: User question
            show_sources: Whether to show source information
            
        Returns:
            Generated answer based on image analysis
        """
        # Search for most relevant images
        relevant_images = self.search_images(question, top_k=3)
        
        if not relevant_images:
            # If CLIP search finds nothing above threshold, analyze first few images
            relevant_images = [(img, 0.0) for img in self.extracted_images[:3]]
        
        # Analyze each relevant image
        analyses = []
        image_sources = []
        for img, score in relevant_images:
            analysis = self.analyze_image_with_vision(img, question)
            analyses.append(f"[Image from Page {img.page_number}]:\n{analysis}")
            image_sources.append((img, score))
        
        # Combine analyses into a response
        combined_analysis = "\n\n---\n\n".join(analyses)
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical AI assistant. Synthesize the image analyses below into a clear, comprehensive answer to the user's question."
                    },
                    {
                        "role": "user",
                        "content": f"Based on these medical image analyses:\n\n{combined_analysis}\n\nQuestion: {question}\n\nProvide a synthesized answer:"
                    }
                ],
                model=self.llm_model,
                temperature=0.3,
                max_tokens=1500
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Image Analysis Results:\n\n{combined_analysis}"
        
        if show_sources:
            sources = "\n\n📚 **Sources:**\n"
            for i, (img, score) in enumerate(image_sources, 1):
                page_label = f"Page {img.page_number}" if img.page_number > 0 else "Uploaded Image"
                sources += f"  [Image {i}] {page_label}\n"
            answer += sources
        
        return answer
    
    def query_with_images(self, question: str, show_sources: bool = True, include_image_analysis: bool = True, top_k_images: int = 2) -> Tuple[str, List[Tuple[ExtractedImage, str]]]:
        """
        Full RAG pipeline with image analysis: Retrieve -> Rerank -> Search Images -> Analyze -> Generate
        Automatically handles text-only, image-only, and mixed documents.
        
        Args:
            question: User question
            show_sources: Whether to show source information
            include_image_analysis: Whether to include relevant image analysis
            top_k_images: Number of relevant images to analyze
            
        Returns:
            Tuple of (generated answer, list of (image, analysis) tuples)
        """
        if not self.is_loaded:
            return "Please load a PDF document or images first.", []
        
        # Retrieve relevant text chunks (only if text index exists)
        chunks = []
        if self.vectorstore and self.chunk_texts:
            chunks = self.retrieve(question)
        
        # Search for relevant images
        image_results = []
        image_context = ""
        
        if include_image_analysis and self.extracted_images and self.enable_image_support:
            relevant_images = self.search_images(question, top_k=top_k_images)
            
            # If CLIP search finds nothing above threshold, use first few images
            if not relevant_images:
                relevant_images = [(img, 0.0) for img in self.extracted_images[:top_k_images]]
            
            for img, score in relevant_images:
                # Analyze each relevant image with the specific question
                analysis = self.analyze_image_with_vision(img, question)
                image_results.append((img, analysis))
                page_label = f"Page {img.page_number}" if img.page_number > 0 else "Uploaded Image"
                image_context += f"\n\n[Image from {page_label} - Similarity: {score:.3f}]\n{analysis}"
        
        if not chunks and not image_results:
            return "No relevant information found in the document.", []
        
        # Build combined prompt with image context
        if image_context:
            answer = self._generate_combined_answer(question, chunks, image_context)
        else:
            answer = self.generate_answer(question, chunks) if chunks else "No text context found."
        
        # Add sources if requested
        if show_sources:
            sources = "\n\n📚 **Sources:**\n"
            if chunks:
                for i, chunk in enumerate(chunks, 1):
                    sources += f"  [Text {i}] Page {chunk.page_number} (relevance: {chunk.score:.3f})\n"
            if image_results:
                for i, (img, _) in enumerate(image_results, 1):
                    sources += f"  [Image {i}] Page {img.page_number}\n"
            answer += sources
        
        return answer, image_results
    
    def _generate_combined_answer(self, query: str, chunks: List[RetrievedChunk], image_context: str) -> str:
        """
        Generate answer combining text chunks and image analysis
        
        Args:
            query: User question
            chunks: Retrieved text chunks
            image_context: Analysis from relevant images
            
        Returns:
            Generated combined answer
        """
        # Build text context
        text_context = ""
        if chunks:
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                context_parts.append(f"[Text Source {i} - Page {chunk.page_number}]\n{chunk.content}")
            text_context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""You are a medical AI assistant with expertise in analyzing both medical text and medical imaging reports.
Answer the user's question based on the provided text context AND image analysis.

INSTRUCTIONS:
1. Synthesize information from BOTH text and image analysis
2. If there's relevant information from images (X-rays, MRI, CT scans, etc.), prioritize mentioning those findings
3. Be specific and cite sources (page numbers, whether from text or image)
4. Provide a comprehensive but concise answer
5. If the answer involves medical findings, present them clearly

TEXT CONTEXT:
{text_context if text_context else "No text context available."}

IMAGE ANALYSIS:
{image_context}

QUESTION: {query}

ANSWER:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical AI assistant that analyzes medical documents and imaging reports. Provide accurate, professional responses based on the provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.llm_model,
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_extracted_images(self) -> List[ExtractedImage]:
        """Get all extracted images from the current document"""
        return self.extracted_images
    
    def get_image_count(self) -> int:
        """Get the number of extracted images"""
        return len(self.extracted_images)


def interactive_session(rag: AdvancedRAGSystem):
    """Run an interactive Q&A session"""
    print("\n" + "="*60)
    print("🤖 ADVANCED RAG SYSTEM - Interactive Mode")
    print("="*60)
    print("Commands:")
    print("  'load <path>'  - Load a new PDF document")
    print("  'quit' or 'exit' - Exit the session")
    print("  Any other text - Ask a question about the document")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("\n📝 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower().startswith('load '):
                pdf_path = user_input[5:].strip()
                try:
                    rag.load_pdf(pdf_path)
                except Exception as e:
                    print(f"❌ Error loading PDF: {e}")
                continue
            
            if not rag.is_loaded:
                print("⚠️ Please load a PDF first: load <path_to_pdf>")
                continue
            
            print("\n🔍 Searching and analyzing...")
            answer = rag.query(user_input)
            print(f"\n🤖 Assistant:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point for CLI mode"""
    import sys
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️ GROQ_API_KEY not found in .env file.")
        api_key = input("Enter your Groq API key: ").strip()
        if not api_key:
            print("❌ API key is required. Exiting.")
            sys.exit(1)
    
    # Initialize RAG system for CLI
    print("\n🚀 Initializing Advanced RAG System...")
    rag = AdvancedRAGSystem(
        groq_api_key=api_key,
        chunk_size=800,
        chunk_overlap=150,
        vector_k=10,
        bm25_k=10,
        final_k=5,
        vector_weight=0.5
    )
    
    # Check if PDF path provided as argument
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        try:
            rag.load_pdf(pdf_path)
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
    
    # Start interactive session
    interactive_session(rag)


if __name__ == "__main__":
    main()