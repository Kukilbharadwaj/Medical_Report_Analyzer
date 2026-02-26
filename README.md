# 🏥 Medical Report Analyzer

An advanced RAG (Retrieval-Augmented Generation) system for analyzing medical reports with multimodal support for both text and medical images (X-rays, MRI, CT scans).

## ✨ Features

- **📄 PDF Processing**: Extract and analyze text from medical reports
- **🖼️ Image Analysis**: Automatic extraction and analysis of medical images using LLaMA Vision
- **📸 Standalone Image Upload**: Upload medical images directly for analysis
- **🔍 Hybrid Search**: Combines vector similarity (FAISS) and keyword search (BM25) 
- **🎯 Smart Reranking**: Cross-encoder reranking for better retrieval accuracy
- **🧠 CLIP Embeddings**: Semantic image search using CLIP
- **💬 Interactive Chat**: User-friendly Streamlit interface

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  PDF Upload  │  │Image Upload  │  │  User Query  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼──────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                              │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ PDF Processor (PyMuPDF)                                │     │
│  │  • Text Extraction → Chunking → Embeddings             │     │
│  │  • Image Extraction (embedded + page rendering)        │     │
│  └────────────────┬──────────────────┬────────────────────┘     │
│                   ▼                  ▼                           │
│  ┌────────────────────────┐  ┌──────────────────────┐          │
│  │   Text Index           │  │   Image Index        │          │
│  │  • FAISS (Vector)      │  │  • CLIP Embeddings   │          │
│  │  • BM25 (Keyword)      │  │  • Base64 Storage    │          │
│  └────────────┬───────────┘  └──────────┬───────────┘          │
└───────────────┼──────────────────────────┼──────────────────────┘
                │                          │
                ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL LAYER                               │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ Hybrid Search                                        │       │
│  │  1. Vector Search (FAISS) → Top-K chunks            │       │
│  │  2. BM25 Search → Top-K chunks                      │       │
│  │  3. Reciprocal Rank Fusion (RRF)                    │       │
│  │  4. Cross-Encoder Reranking → Final chunks          │       │
│  └──────────────────┬───────────────────────────────────┘       │
│                     │                                            │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ Image Search                                         │       │
│  │  1. CLIP Text Embedding from query                  │       │
│  │  2. Cosine Similarity with image embeddings         │       │
│  │  3. Top-K relevant images                           │       │
│  └──────────────────┬───────────────────────────────────┘       │
└────────────────────┼─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GENERATION LAYER                               │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ Context Builder                                      │       │
│  │  • Text chunks from retrieval                        │       │
│  │  • Image analysis from LLaMA Vision                  │       │
│  └──────────────────┬───────────────────────────────────┘       │
│                     ▼                                            │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ LLM (Groq - Llama 3.3 70B)                           │       │
│  │  • Synthesizes text + image context                  │       │
│  │  • Generates comprehensive answer                    │       │
│  │  • Cites sources (pages, images)                     │       │
│  └──────────────────┬───────────────────────────────────┘       │
└────────────────────┼─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                               │
│              📝 Formatted Answer with Sources                    │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Groq (Llama 3.3 70B) |
| **Vision Model** | Groq (LLaMA Scout 17B) |
| **Embeddings** | HuggingFace Transformers (MiniLM) |
| **Image Embeddings** | OpenAI CLIP |
| **Vector Store** | FAISS |
| **Keyword Search** | BM25 (rank-bm25) |
| **Reranking** | Cross-Encoder (MS MARCO) |
| **PDF Processing** | PyMuPDF, PyPDF |
| **UI Framework** | Streamlit |
| **Framework** | LangChain |

## 📦 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Medical_Report_Analyzer
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## 🚀 Usage

### Web Interface (Streamlit)
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Command Line Interface
```bash
python llm.py [path/to/document.pdf]
```

## 📂 Project Structure

```
Medical_Report_Analyzer/
│
├── app.py                 # Streamlit web interface
├── llm.py                 # Core RAG system implementation
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
└── README.md             # Project documentation
```

## 🎯 How It Works

### 1. **Document Loading**
- Extracts text using PyPDF
- Extracts embedded images and renders image-only pages
- Handles text-only, image-only, or mixed PDFs

### 2. **Indexing**
- **Text**: Split into chunks → Generate embeddings → Store in FAISS + BM25
- **Images**: Generate CLIP embeddings → Store with base64 data

### 3. **Query Processing**
- Parse user question
- Retrieve relevant text chunks (Hybrid Search + Reranking)
- Retrieve relevant images (CLIP similarity)
- Analyze images with LLaMA Vision

### 4. **Answer Generation**
- Combine text and image context
- Generate comprehensive answer using Groq LLM
- Cite sources (page numbers, images)

## 📊 Key Features Explained

### Hybrid Search
Combines semantic understanding (vector search) with keyword matching (BM25) using Reciprocal Rank Fusion for optimal results.

### Cross-Encoder Reranking
After initial retrieval, a cross-encoder model re-scores candidates by considering the actual interaction between query and document.

### Multimodal Analysis
- **CLIP**: Semantic image search matching query intent
- **LLaMA Vision**: Deep analysis of X-rays, MRI, CT scans
- **Synthesis**: Combines insights from both text and images

### Automatic Fallback
- Text-only PDFs → Text retrieval
- Image-only PDFs → Image analysis
- Mixed PDFs → Multimodal analysis

## 🔑 API Keys

Get your Groq API key from [Groq Console](https://console.groq.com/)

## 📝 Example Queries

- "What is the diagnosis from this report?"
- "Analyze the X-ray findings"
- "Summarize the key medical findings"
- "What abnormalities are visible in the MRI?"

## ⚠️ Notes

- Minimum Python version: 3.8+
- First run downloads model weights (~500MB)
- Requires internet connection for API calls
- Supports PDF, PNG, JPG, JPEG, BMP, TIFF, WEBP

## 📄 License

MIT License

---

