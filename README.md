# 📊 RAG Research Assistent

A sophisticated AI-powered literature review system that automatically retrieves, analyzes, and synthesizes academic research papers using advanced RAG (Retrieval-Augmented Generation) technology.


## 🌟 Features

### 🔍 Intelligent Paper Retrieval
- **arXiv Search**: Smart query expansion and topic matching
- **Temporal Filtering**: Search papers from 1990 to present day (configurable)
- **Relevance Scoring**: AI-powered relevance assessment (0-1 scoring system)
- **Automatic Deduplication**: Avoid duplicate papers across queries
- **Caching**: 24-hour cache to avoid redundant API calls

### 📊 Comprehensive Analysis
- **AI-Powered Literature Reviews**: Generate detailed academic reviews using local LLMs
- **Publication Trend Analysis**: Visualize research trends over decades
- **Relevance Distribution**: Interactive histograms and statistics
- **Interactive Data Tables**: Edit and explore paper metadata in real-time

### 💾 Professional Export
- **Multiple Formats**: Markdown, PDF, BibTeX, CSV, and JSON
- **Academic Citations**: Automatic BibTeX citation generation
- **Research Reports**: Professional PDF reports with proper academic formatting
- **Structured Data**: Complete dataset export for further analysis

### 🚀 Technical Excellence
- **Local Processing**: No API costs, complete privacy and data security
- **Vector Search**: ChromaDB-powered semantic similarity search
- **Advanced Prompt Engineering**: Sophisticated LLM prompts for quality reviews
- **Error Resilience**: Comprehensive error handling and graceful fallbacks

## 📸 Demo

| Literature Review | Trend Analysis | Paper Management |
|-------------------|----------------|------------------|
| *AI-generated comprehensive reviews* | *Publication trends visualization* | *Interactive paper exploration* |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed locally
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/BBhushan1/RAG_Research.git
cd RAG_Research
```

3. **Set up Ollama**
```bash
# Install Ollama 

# Pull a model
ollama pull gemma3:4b
```

4. **Configure environment**
```bash
# Edit .env with your preferences
```

5. **Launch the application**
```bash
streamlit run app.py
```

## ⚙️ Configuration

Edit the `.env` file to customize:

```env
LLM_MODEL=gemma3:4b
OLLAMA_URL=http://localhost:11434/api/generate
MAX_RESULTS=10
START_YEAR=1990
END_YEAR=2025
OUTPUT_DIR=./output
VECTOR_DB_PATH=./vector_db
```

## 🎯 Usage

1. **Enter your research query** in the text area
2. **Click "Search & Analyze Papers"** to retrieve relevant research
3. **Explore the results** across four interactive tabs:
   - 📚 Literature Review: AI-generated comprehensive analysis
   - 📊 Analysis: Trends and relevance scoring
   - 📄 Papers: Detailed paper information
   - 💾 Export: Download results in multiple formats

### Example Queries
- "Transformer architecture improvements in NLP"
- "Few-shot learning techniques for computer vision"
- "Neural network compression methods"
- "Self-supervised learning approaches 2020-2024"

## 🏗️ Project Architecture

```
academic-rag-assistant/
├── main.py                 # Main Streamlit application
├── llm_generation.py      # LLM prompt engineering
├── arxiv_retriever.py     # arXiv paper retrieval
├── chroma_db.py          # Vector database
├── embedder.py           # Text embedding generation
├── config.py             # Configuration management
└── README.md            # Project documentation
```

### Technical Stack
- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **LLM**: Ollama (local models)
- **Embeddings**: Sentence Transformers
- **Data Source**: arXiv API
- **PDF Generation**: ReportLab

## 🔧 Advanced Features

### Smart Prompt Engineering
- Dynamic prompt selection based on query complexity
- Academic-style formatting and critical analysis
- Context-aware paper selection for LLM constraints

### Intelligent Caching
- 24-hour cache for repeated queries
- Smart cache invalidation
- Performance optimization for frequent searches

### Quality Assurance
- Input validation and sanitization
- Comprehensive error handling
- Fallback content generation


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **arXiv** for providing access to research papers
- **Ollama** for the excellent local LLM framework
- **ChromaDB** for vector database capabilities
- **Streamlit** for the interactive web framework


**Built with ❤️ for the AI research community**
