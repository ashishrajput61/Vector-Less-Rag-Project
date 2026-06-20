# 🚀 Vector-Less RAG

A Retrieval-Augmented Generation (RAG) application that retrieves relevant information from documents **without relying on vector databases or embeddings**.

This project explores an alternative retrieval architecture where document understanding and context selection are performed using intelligent document structuring, keyword-based retrieval, and LLM-guided reasoning instead of traditional vector similarity search.

## 🌐 Live Demo

**Try it here:** https://vector-less-rag-project-by-ashishrajput.streamlit.app/

---

## 📌 Overview

Traditional RAG systems typically depend on:

* Document chunking
* Embedding generation
* Vector databases
* Similarity search

This project demonstrates a **Vector-Less RAG** approach that eliminates the need for embedding generation and vector storage while still enabling efficient document retrieval and question answering.

The system focuses on:

* Lower infrastructure complexity
* Reduced operational cost
* Faster setup and deployment
* Transparent retrieval logic
* Improved explainability

---

## ✨ Features

* 📄 Upload and process documents
* 🔍 Intelligent document retrieval without vector databases
* 🤖 LLM-powered question answering
* ⚡ Lightweight architecture
* 📊 Context-aware response generation
* 🎯 Reduced dependency on embedding models
* 🌐 Interactive Streamlit interface

---

## 🏗️ Architecture

```text
User Query
     │
     ▼
Document Processing
     │
     ▼
Structured Index Creation
     │
     ▼
Query Analysis
     │
     ▼
Relevant Context Retrieval
     │
     ▼
LLM Response Generation
     │
     ▼
Final Answer
```

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend

* Python

### AI Components

* Large Language Models (LLMs)
* Document Processing Pipeline
* Retrieval Logic

### Libraries

* Streamlit
* LangChain (if used)
* Google Gemini / OpenAI API (if used)
* Pandas
* PyPDF

---

## 📂 Project Structure

```bash
Vector-Less-RAG/
│
├── app.py
├── requirements.txt
├── data/
├── utils/
├── retrieval/
├── document_processing/
├── assets/
└── README.md
```

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/vector-less-rag.git
cd vector-less-rag
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
API_KEY=your_api_key
```

If using Gemini:

```env
GOOGLE_API_KEY=your_google_api_key
```

If using OpenAI:

```env
OPENAI_API_KEY=your_openai_api_key
```

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

Application will be available at:

```text
http://localhost:8501
```

---

## 📖 How It Works

1. Upload a document.
2. The system processes and structures the content.
3. A retrieval mechanism identifies the most relevant sections.
4. Selected context is passed to the LLM.
5. The model generates an accurate response grounded in retrieved content.

Unlike traditional RAG pipelines, no vector database is required.

---

## 🎯 Advantages of Vector-Less RAG

* No embedding generation cost
* No vector database maintenance
* Easier debugging
* Faster prototyping
* Lower infrastructure requirements
* Better transparency in retrieval decisions

---

## 📸 Screenshots

### Home Page

Add screenshot here:

```text
assets/home.png
```

### Question Answering

Add screenshot here:

```text
assets/chat.png
```

---

## 🔮 Future Improvements

* Multi-document support
* Hybrid retrieval strategies
* Citation generation
* Conversation memory
* Advanced reranking
* Evaluation dashboard

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Ashish Rajput**

* GitHub: https://github.com/ashishrajput61/
* LinkedIn: https://www.linkedin.com/in/ashish-r-7a5529268/

---

⭐ If you found this project useful, consider giving it a star on GitHub!
