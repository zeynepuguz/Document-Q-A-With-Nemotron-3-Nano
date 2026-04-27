# 📄 Document Q&A with Nemotron 3 Nano

This project is a high-performance **RAG (Retrieval-Augmented Generation)** application that allows you to chat with your private documents using the **Nemotron 3 Nano (30B)** model via Ollama.

---

## 🚀 Features

- **Multi-Format Support:** PDF, Markdown, TXT, JSON, YAML, and more  
- **Smart Retrieval:** Selects the most relevant document segments  
- **Citation System:** Provides references in `[Dxx:Syyy]` format  
- **Advanced Controls:** Adjust Temperature, Max Tokens, and Context Window  
- **Flexible Ingestion:** Upload files or read from a local folder  

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/nemotron-doc-qa.git
cd nemotron-doc-qa

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/nemotron-doc-qa.git
cd nemotron-doc-qa
```

### 2. Install dependencies

```bash
pip install streamlit pymupdf ollama python-dotenv
```

### 3. Setup environment variables

```env
OLLAMA_API_KEY=your_actual_api_key_here
```

---

## 💻 Usage

```bash
streamlit run app.py
```

---

## 📸 Screenshots

![App Screenshot](./assets/image.png)

![App Screenshot 2](./assets/image1.png)

---
---

## 🧠 Tech Stack

* Streamlit
* PyMuPDF (fitz)
* Ollama (Nemotron 3 Nano)
* Python

---
