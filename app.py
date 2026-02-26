"""
Streamlit UI for Advanced RAG System with Medical Image Analysis
"""

import os
import streamlit as st
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from llm import AdvancedRAGSystem

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Medical Report Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'rag' not in st.session_state:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error(
                "⚠️ GROQ_API_KEY not found in environment variables. "
                "Please create a .env file with your API key."
            )
            st.stop()
        
        # Initialize RAG system with image support
        st.session_state.rag = AdvancedRAGSystem(
            groq_api_key=api_key,
            chunk_size=800,
            chunk_overlap=150,
            vector_k=10,
            bm25_k=10,
            final_k=5,
            vector_weight=0.5,
            enable_image_support=True
        )
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'pdf_loaded' not in st.session_state:
        st.session_state.pdf_loaded = False
    
    if 'pdf_name' not in st.session_state:
        st.session_state.pdf_name = ""
    
    if 'image_count' not in st.session_state:
        st.session_state.image_count = 0
    
    if 'use_image_analysis' not in st.session_state:
        st.session_state.use_image_analysis = True
    
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    
    if 'has_content' not in st.session_state:
        st.session_state.has_content = False


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.title("🏥 Medical Report Analyzer")
    
    st.markdown("""
    **Features:**
    - 📄 PDF document processing with text analysis
    - 🖼️ Medical image extraction and analysis (X-rays, MRI, CT scans)
    - � Direct image upload for standalone analysis
    - �🔍 Hybrid search (Vector + BM25)
    - 🎯 Cross-encoder reranking
    - 🧠 CLIP embeddings for image search
    - 👁️ LLaMA Vision for medical image analysis
    - 💬 Powered by Groq's Llama model
    """)
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📤 Upload Medical Files")
        
        # Tab selection for upload type
        upload_tab = st.radio(
            "Upload type:",
            ["📄 PDF Report", "🖼️ Medical Images"],
            horizontal=True
        )
        
        if upload_tab == "📄 PDF Report":
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=['pdf'],
                help="Upload a medical report PDF to analyze (supports text + images)"
            )
            
            if uploaded_file is not None:
                if st.button("🚀 Process PDF", type="primary", use_container_width=True):
                    with st.spinner("Processing PDF (extracting text and images)..."):
                        try:
                            # Save uploaded file to temporary location
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                            
                            # Load the PDF
                            num_chunks = st.session_state.rag.load_pdf(tmp_path)
                            
                            # Get image count
                            image_count = st.session_state.rag.get_image_count()
                            
                            # Update session state
                            st.session_state.pdf_loaded = True
                            st.session_state.pdf_name = uploaded_file.name
                            st.session_state.image_count = image_count
                            st.session_state.has_content = True
                            st.session_state.uploaded_images = []
                            
                            # Clean up temp file
                            try:
                                Path(tmp_path).unlink()
                            except Exception:
                                pass
                            
                            if num_chunks > 0:
                                st.success(
                                    f"✅ PDF loaded successfully!\n\n"
                                    f"📄 File: {uploaded_file.name}\n\n"
                                    f"📊 Created {num_chunks} text chunks\n\n"
                                    f"🖼️ Extracted {image_count} images\n\n"
                                    f"💡 You can now ask questions!"
                                )
                            else:
                                st.success(
                                    f"✅ Image-only PDF loaded!\n\n"
                                    f"📄 File: {uploaded_file.name}\n\n"
                                    f"📊 No text found (scanned/image PDF)\n\n"
                                    f"🖼️ Extracted {image_count} images\n\n"
                                    f"💡 Images will be analyzed with LLaMA Vision!"
                                )
                        except Exception as e:
                            st.error(f"❌ Error loading PDF: {str(e)}")
        
        else:  # Image upload tab
            uploaded_images = st.file_uploader(
                "Choose medical images",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
                accept_multiple_files=True,
                help="Upload medical images (X-rays, MRI, CT scans, etc.) for analysis"
            )
            
            if uploaded_images:
                if st.button("🚀 Process Images", type="primary", use_container_width=True):
                    with st.spinner("Processing images..."):
                        try:
                            image_files = [
                                {'data': img.getvalue(), 'name': img.name}
                                for img in uploaded_images
                            ]
                            count = st.session_state.rag.load_images(image_files)
                            
                            st.session_state.image_count = st.session_state.rag.get_image_count()
                            st.session_state.has_content = True
                            st.session_state.uploaded_images = uploaded_images
                            
                            st.success(
                                f"✅ {count} image(s) loaded!\n\n"
                                f"🖼️ Total images: {st.session_state.image_count}\n\n"
                                f"💡 Ask questions about the images!"
                            )
                        except Exception as e:
                            st.error(f"❌ Error loading images: {str(e)}")
        
        # Display status
        st.divider()
        if st.session_state.pdf_loaded or st.session_state.has_content:
            if st.session_state.pdf_name:
                st.success(f"📄 Document: {st.session_state.pdf_name}")
            if st.session_state.image_count > 0:
                st.info(f"🖼️ Images available: {st.session_state.image_count}")
            
            # Image analysis toggle
            st.session_state.use_image_analysis = st.checkbox(
                "🔬 Include image analysis",
                value=st.session_state.use_image_analysis,
                help="When enabled, relevant medical images will be analyzed using LLaMA Vision"
            )
        else:
            st.info("👆 Upload a PDF or images to get started")
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.markdown("**Note:** Your Groq API key is loaded from the `.env` file.")
    
    # Main chat interface
    st.header("💬 Ask Questions About Your Medical Report")
    
    # Display extracted images section if available
    if (st.session_state.pdf_loaded or st.session_state.has_content) and st.session_state.image_count > 0:
        with st.expander(f"🖼️ View Extracted/Uploaded Medical Images ({st.session_state.image_count})", expanded=False):
            images = st.session_state.rag.get_extracted_images()
            if images:
                cols = st.columns(min(3, len(images)))
                for i, img in enumerate(images):
                    with cols[i % 3]:
                        caption = f"Page {img.page_number}, Image {img.image_index + 1}" if img.page_number > 0 else f"Uploaded Image {img.image_index + 1}"
                        st.image(img.image, caption=caption, use_container_width=True)
    
    # Display example questions if no messages
    if len(st.session_state.messages) == 0:
        st.markdown("### Example Questions:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 What is the diagnosis?", use_container_width=True):
                if st.session_state.pdf_loaded or st.session_state.has_content:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": "What is the diagnosis from this medical report?"
                    })
                    st.rerun()
        
        with col2:
            if st.button("🖼️ Analyze the X-ray/MRI", use_container_width=True):
                if st.session_state.pdf_loaded or st.session_state.has_content:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": "Can you analyze the medical images (X-ray, MRI, or CT scan) in this report?"
                    })
                    st.rerun()
        
        with col3:
            if st.button("📋 Summarize findings", use_container_width=True):
                if st.session_state.pdf_loaded or st.session_state.has_content:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": "Summarize the key medical findings from this report including any imaging results."
                    })
                    st.rerun()
        
        st.divider()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display associated images if any
            if "images" in message and message["images"]:
                st.markdown("**📸 Analyzed Medical Images:**")
                for img, analysis in message["images"]:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        caption = f"Page {img.page_number}" if img.page_number > 0 else "Uploaded Image"
                        st.image(img.image, caption=caption, use_container_width=True)
                    with col2:
                        with st.expander("View Image Analysis", expanded=False):
                            st.markdown(analysis)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the medical report or images..."):
        if not (st.session_state.pdf_loaded or st.session_state.has_content):
            st.warning("⚠️ Please upload a PDF document or images first.")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document and images..."):
                    try:
                        # Use image-aware query if enabled and images exist
                        if st.session_state.use_image_analysis and st.session_state.image_count > 0:
                            answer, image_results = st.session_state.rag.query_with_images(
                                prompt, 
                                show_sources=True,
                                include_image_analysis=True,
                                top_k_images=2
                            )
                            st.markdown(answer)
                            
                            # Display analyzed images
                            if image_results:
                                st.markdown("**📸 Analyzed Medical Images:**")
                                for img, analysis in image_results:
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        caption = f"Page {img.page_number}" if img.page_number > 0 else "Uploaded Image"
                                        st.image(img.image, caption=caption, use_container_width=True)
                                    with col2:
                                        with st.expander("View Image Analysis", expanded=True):
                                            st.markdown(analysis)
                            
                            # Store message with images
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": answer,
                                "images": image_results
                            })
                        else:
                            # Text-only query
                            answer = st.session_state.rag.query(prompt, show_sources=True)
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
