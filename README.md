# 📄 Document Q&A with Nemotron 3 Nano

This project is a high-performance **RAG (Retrieval-Augmented Generation)** application that allows you to chat with your private documents using the **Nemotron 3 Nano (30B)** model via Ollama.

---

## 🚀 Features

* **Multi-Format Support:** PDF, Markdown, TXT, JSON, YAML ve daha fazlası
* **Smart Retrieval:** En alakalı segmentleri seçer
* **Citation System:** `[Dxx:Syyy]` formatında referans verir
* **Advanced Controls:** Temperature, Max Tokens, Context Window ayarları
* **Flexible Ingestion:** Dosya yükleme veya klasör okuma

---

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

Proje kök dizininde `.env` dosyası oluştur:

```env
OLLAMA_API_KEY=your_actual_api_key_here
```

---

## 💻 Usage

Uygulamayı başlat:

```bash
streamlit run app.py
```

---

## 📸 Screenshots

![App Screenshot](./assets/image.png)

![App Screenshot 2](./assets/image1.png)

---

## ⚠️ Important Notes

* Eğer PDF **taranmış (image-based)** ise OCR gerekir (Tesseract)
* Smart mode keyword eşleşmesine göre çalışır
* Model sadece dokümandan cevap verir, yoksa `"I don't know..."` döner

---

## 🧠 Tech Stack

* Streamlit
* PyMuPDF (fitz)
* Ollama (Nemotron 3 Nano)
* Python

---
