# GenAI Assistant for Software Architects

A Streamlit web application that uses AI to analyze changes between software requirements and architecture documents, providing insights and recommendations for architects.

## Features

- Upload and analyze Software Requirements Specification (SRS) documents
- Upload and analyze Software Architecture Specification (SAS) documents
- Track and analyze requirement changes
- Get AI-generated recommendations on architecture updates based on requirement changes
- Supports PDF, DOCX, and TXT file formats

## Prerequisites

- Python 3.13 or higher
- Poetry (Python package manager)
- OpenAI API key

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/genAI-for-Architects.git
cd genAI-for-Architects
```

### 2. Install Poetry

If you don't have Poetry installed, install it by following the instructions at [Python Poetry Installation](https://python-poetry.org/docs/#installation).

For Windows:
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

For macOS/Linux:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. Create .env file with OpenAI API key

Create a file named `.env` in the root directory of the project:

```powershell
New-Item .env -ItemType File
```

Add your OpenAI API key to this file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Activate Poetry shell

```powershell
poetry shell
```

### 5. Install dependencies

```powershell
poetry install
```

## Running the Application

Start the Streamlit application:

```powershell
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

## How to Use

1. **Upload Documents**:
   - Upload your Software Requirements Specification (SRS) document using the first file uploader
   - Upload your Software Architecture Specification (SAS) document using the second file uploader

2. **Edit Requirements**:
   - Make changes to the requirements in the text area if needed

3. **Analyze Changes**:
   - Click the "Analyze Changes" button to generate AI recommendations
   - View the detected additions, removals, and modifications
   - Review the AI analysis and architecture update recommendations

## Technologies Used

- Streamlit - Web application framework
- LangChain - Framework for building applications with LLMs
- OpenAI - GPT models for analysis
- FAISS - Vector database for efficient similarity search
- HuggingFace - Embeddings and models
- PyMuPDF & python-docx - Document parsing

## License
-

## Author

Ravindra Shinde