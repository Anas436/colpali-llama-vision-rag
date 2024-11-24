import streamlit as st
import torch
from PIL import Image
import sqlite3
import numpy as np
import pickle
import base64
import io
from colpali_engine.models import ColQwen2, ColQwen2Processor
import gc
from pdf2image import convert_from_bytes
from io import BytesIO
import hashlib  # Import hashlib for hashing

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device_map = get_device()

# Function to load the model and processor
@st.cache_resource
def load_model():
    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v0.1",
        torch_dtype=torch.bfloat16,
        device_map=device_map  # Use "mps" if on Apple Silicon; otherwise, use "cpu" or "cuda"
    )
    
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")
    
    return model, processor

# Function to get a database connection
def get_db_connection():
    conn = sqlite3.connect('image_embeddings.db')
    return conn

def process_and_index_image(image, img_str, image_hash, processor, model):
    # Store in database
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_base64 TEXT,
            image_hash TEXT UNIQUE,
            embedding BLOB
        )
    ''')
    # Check if the image hash already exists
    c.execute('SELECT id FROM embeddings WHERE image_hash = ?', (image_hash,))
    result = c.fetchone()
    if result:
        # Image already indexed
        conn.close()
        return
    # Process image to get embedding
    batch_images = processor.process_images([image]).to(model.device)
    with torch.no_grad():
        image_embeddings = model(**batch_images)
    image_embedding = image_embeddings[0].cpu().to(torch.float32).numpy()
    # Serialize the embedding
    embedding_bytes = pickle.dumps(image_embedding)
    c.execute('INSERT INTO embeddings (image_base64, image_hash, embedding) VALUES (?, ?, ?)', (img_str, image_hash, embedding_bytes))
    conn.commit()
    conn.close()

def clear_cache():
    """Clear GPU memory cache for different platforms."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        # CPU doesn't need explicit cache clearing
    except Exception as e:
        print(f"Warning: Could not clear cache: {str(e)}")

def main():
    st.title("📷 Image RAG(Colpali + Llama Vision)")

    model, processor = load_model()

    # Initialize session state for image hashes
    if 'image_hashes' not in st.session_state:
        st.session_state.image_hashes = set()

    # Use st.radio for tab selection
    tab = st.radio("Navigation", ["➕ Add to Index", "🔍 Query Index"])

    if tab == "➕ Add to Index":
        st.header("Add Images to Index")
        # File uploader
        uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'pdf'])
        if uploaded_files:
            # Process the uploaded images
            for uploaded_file in uploaded_files:
                if uploaded_file.type == 'application/pdf':
                    images = convert_from_bytes(uploaded_file.read())
                    for image in images:
                        buffer = BytesIO()
                        image.save(buffer, format="PNG")
                        byte_data = buffer.getvalue()
                        img_str = base64.b64encode(byte_data).decode('utf-8')
                        # Compute image hash
                        image_hash = hashlib.sha256(byte_data).hexdigest()
                        if image_hash in st.session_state.image_hashes:
                            continue  # Skip if already processed in this session
                        process_and_index_image(image, img_str, image_hash, processor, model)
                        st.session_state.image_hashes.add(image_hash)
                else:
                    # Read image data
                    image_data = uploaded_file.read()
                    # Compute image hash
                    image_hash = hashlib.sha256(image_data).hexdigest()
                    if image_hash in st.session_state.image_hashes:
                        continue  # Skip if already processed in this session
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_data)).convert('RGB')
                    # Encode image to base64
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    process_and_index_image(image, img_str, image_hash, processor, model)
                    st.session_state.image_hashes.add(image_hash)
            st.success("Images added to index.")

    elif tab == "🔍 Query Index":
        st.header("Query Index")
        query = st.text_input("Enter your query")
        if query:
            # Process query
            with torch.no_grad():
                batch_query = processor.process_queries([query]).to(model.device)
                query_embedding = model(**batch_query)
            query_embedding_cpu = query_embedding.cpu().to(torch.float32).numpy()[0]

            # Retrieve image embeddings from database
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('SELECT image_base64, embedding FROM embeddings')
            rows = c.fetchall()
            conn.close()

            if not rows:
                st.warning("No images found in the index. Please add images first.")
                return

            # Set fixed sequence length
            fixed_seq_len = 620  # Adjust based on your embeddings

            image_embeddings_list = []
            image_base64_list = []

            for row in rows:
                image_base64, embedding_bytes = row
                embedding = pickle.loads(embedding_bytes)
                seq_len, embedding_dim = embedding.shape

                # Adjust to fixed sequence length
                if seq_len < fixed_seq_len:
                    padding = np.zeros((fixed_seq_len - seq_len, embedding_dim), dtype=embedding.dtype)
                    embedding_fixed = np.concatenate([embedding, padding], axis=0)
                elif seq_len > fixed_seq_len:
                    embedding_fixed = embedding[:fixed_seq_len, :]
                else:
                    embedding_fixed = embedding  # No adjustment needed

                image_embeddings_list.append(embedding_fixed)
                image_base64_list.append(image_base64)

            # Stack embeddings
            retrieved_image_embeddings = np.stack(image_embeddings_list)

            # Adjust query embedding
            seq_len_q, embedding_dim_q = query_embedding_cpu.shape

            if seq_len_q < fixed_seq_len:
                padding = np.zeros((fixed_seq_len - seq_len_q, embedding_dim_q), dtype=query_embedding_cpu.dtype)
                query_embedding_fixed = np.concatenate([query_embedding_cpu, padding], axis=0)
            elif seq_len_q > fixed_seq_len:
                query_embedding_fixed = query_embedding_cpu[:fixed_seq_len, :]
            else:
                query_embedding_fixed = query_embedding_cpu

            # Convert to tensors
            query_embedding_tensor = torch.from_numpy(query_embedding_fixed).to(model.device).unsqueeze(0)
            retrieved_image_embeddings_tensor = torch.from_numpy(retrieved_image_embeddings).to(model.device)

            # Compute similarity scores
            with torch.no_grad():
                scores = processor.score_multi_vector(query_embedding_tensor, retrieved_image_embeddings_tensor)
            scores_np = scores.cpu().numpy().flatten()
            del query_embedding_tensor, retrieved_image_embeddings_tensor, scores  # Free up memory
            clear_cache()

            # Combine images and scores
            similarities = list(zip(image_base64_list, scores_np))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            if similarities:
                st.write("Most similar image:")
                img_str, score = similarities[0]
                st.write(f"Similarity Score: {score:.4f}")
                # Decode image from base64
                img_data = base64.b64decode(img_str)
                image = Image.open(io.BytesIO(img_data))
                st.image(image)
            else:
                st.write("No similar images found.")

            st.write("AI Response:")

            import ollama

            response_container = st.empty()
            

            # Spinner only for the initial API call
            
            
            stream = ollama.chat(
                model="llama3.2-vision",
                messages=[
                    {
                        'role': 'user',
                        'content': "Please answer the following question using only the information visible in the provided image" 
                        " Do not use any of your own knowledge, training data, or external sources."
                        " Base your response solely on the content depicted within the image."
                        " If there is no relation with question and image," 
                        f" you can respond with 'Question is not related to image'.\nHere is the question: {query}",
                        'images': [img_data]
                    }
                ],
                stream=True
            )

            

            collected_chunks = []
            stream_iter = iter(stream)

            with st.spinner('⏳ Generating Response...'):
                try:
                    # Get the first chunk
                    first_chunk = next(stream_iter)
                    chunk_content = first_chunk['message']['content']
                    collected_chunks.append(chunk_content)
                    # Display the initial response
                    complete_response = ''.join(collected_chunks)
                    response_container.markdown(complete_response)
                except StopIteration:
                    # Handle if no chunks are received
                    pass

            # Continue streaming the rest of the response
            for chunk in stream_iter:
                chunk_content = chunk['message']['content']
                collected_chunks.append(chunk_content)
                complete_response = ''.join(collected_chunks)
                response_container.markdown(complete_response)


            clear_cache()
            gc.collect()

if __name__ == "__main__":
    main()
