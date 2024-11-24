# 🖼️ Image RAG (Colpali + LLaMA Vision)

A powerful Retrieval-Augmented Generation (RAG) system combining Colpali's ColQwen image embeddings with LLaMA Vision via Ollama.

## 🌟 Key Features

- 🧬 ColQwen model for generating powerful image embeddings via Colpali
- 🤖 LLaMA Vision integration through Ollama for image understanding
- 📥 Intelligent image indexing with duplicate detection
- 💬 Natural language image queries
- 📄 PDF document support
- 🔍 Semantic similarity search
- 📊 Efficient SQLite storage

## 🛠️ Technical Stack

- **Embedding Model**: ColQwen via Colpali
- **Vision Model**: LLaMA Vision via Ollama
- **Frontend**: Streamlit
- **Database**: SQLite
- **Image Processing**: Pillow, pdf2image
- **ML Framework**: PyTorch


## ⚡ Quick Start

1. Install Poppler (required for PDF support):

   **Mac:**
   ```bash
   brew install poppler
   ```

   **Windows:**
   1. Download the latest poppler package from: https://github.com/oschwartz10612/poppler-windows/releases/
   2. Extract the downloaded zip to a location (e.g., `C:\Program Files\poppler`)
   3. Add bin directory to PATH:
      - Open System Properties > Advanced > Environment Variables
      - Under System Variables, find and select "Path"
      - Click "Edit" > "New"
      - Add the bin path (e.g., `C:\Program Files\poppler\bin`)
   4. Verify installation:
      ```bash
      pdftoppm -h
      ```

2. Clone and setup environment:
   ```bash
   git clone https://github.com/kturung/colpali-llama-vision-rag.git
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   # or
   .\venv\Scripts\activate  # For Windows
   pip install -r requirements.txt
   ```

3. Install Ollama from https://ollama.com

4. Launch application:
   ```bash
   streamlit run app.py
   ```

> Note: Restart your terminal/IDE after modifying PATH variables


## 💡 Usage

### 📤 Adding Images
1. Navigate to "➕ Add to Index"
2. Upload images/PDFs
3. System automatically:
   - Generates ColQwen embeddings
   - Checks for duplicates
   - Stores in SQLite

### 🔎 Querying
1. Go to "🔍 Query Index"
2. Enter natural language query
3. View similar images
4. Get LLaMA Vision analysis


## 💾 Database Schema

```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_base64 TEXT,
    image_hash TEXT UNIQUE,
    embedding BLOB
)
```
