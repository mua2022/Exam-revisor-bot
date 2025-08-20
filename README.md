# 📚 DocGenius RAG Chatbot  

DocGenius is an AI-powered **Retrieval-Augmented Generation (RAG) chatbot** built with **Streamlit, FAISS, LangChain, and Groq LLM API**.  
It allows you to upload documents (PDFs, text, Markdown) or fetch from URLs, and then ask natural language questions.  

This makes it a perfect tool for:  
- 🎓 **Exam Revision** – Upload lecture notes, research papers, or textbooks and ask focused questions to quickly revise.  
- 💼 **Executive Summaries** – Upload reports, board papers, or meeting notes and get concise summaries and insights.  

---
You can check out the deployed site by clicking the following :[Exam-revisor-bot](https://exam-revisor-bot-abqkojew7o64xpwhls3c85.streamlit.app/)

---

## 🚀 Features  

- 📂 Upload multiple PDFs, TXT, or MD files  
- 🌐 Ingest content from URLs  
- ⚡ Vector search using **FAISS**  
- 🧠 RAG pipeline powered by **LangChain + Groq**  
- 💬 Interactive chat interface with styled chat bubbles  
- 📜 Persistent chat history during session  
- ✅ Rebuild index with new data anytime  

---

## 🛠️ Tech Stack  

- [Streamlit](https://streamlit.io/) – frontend UI  
- [LangChain](https://www.langchain.com/) – RAG framework  
- [FAISS](https://faiss.ai/) – vector store for embeddings  
- [Groq API](https://groq.com/) – lightning-fast LLM inference  
- [ChromaDB](https://www.trychroma.com/) – optional document storage  

---

## 📦 Installation  

Clone the repo:
```bash
git clone https://github.com/mua2022/RAG_Document_Analyser.git
cd RAG_Document_Analyser
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:
```bash
streamlit run app.py
```

---

## 🎯 Usage  

1. Upload documents or paste URLs  
2. Click **Build/Rebuild Index**  
3. Start asking questions in natural language  
4. View answers in real time with **chat bubbles**  

---

## 📌 Example Use Cases  

- **For Students**  
  - Upload lecture slides and notes  
  - Ask *“Summarize Chapter 5 in simple terms”*  
  - Ask *“What are the key differences between TCP and UDP?”*  

- **For Executives**  
  - Upload a 50-page report  
  - Ask *“Give me the top 3 insights for the meeting”*  
  - Ask *“Summarize the financial risks mentioned”*  

---

## 🤝 Contributing  

Pull requests are welcome! For major changes, open an issue first to discuss your idea.  

---

## 📜 License  

MIT License – free to use, modify, and distribute.  

---

💡 *DocGenius makes documents talk back – whether for exams or the boardroom.*  
