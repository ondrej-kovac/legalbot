# LegalBot - A Legal Document Assistant (LLM + FAISS + PDF)

A lightweight command-line legal assistant that allows users to load legal PDF documents, index them using a FAISS vector store, and ask questions that are answered by a locally-run LLM.

**IMPORTANT NOTE:** This is a simple portfolio project and should not be relied on to examine real legal documents. It uses very simple models that can be run locally and often makes mistakes. 

## Features

- CLI interface
- Loads PDF documents
- Builds or loads a FAISS vector store
- Gives contextual answers to user's questions using an LLM

## Tech Stack

- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [Llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Hugging Face Hub](https://huggingface.co/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)

## Default Models

- Embedding model: [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)
- LLM: [Mistral-7B-Instruct-v0.2.Q4_K_M-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf)


## Installation

1. **Clone or Download the Repository**

```
git clone https://github.com/ondrej-kovac/legalbot.git
cd legalbot
```
Or just download the ZIP from GitHub and extract it.

2. **Create a Virtual Environment (recommended)**
```
python -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```
pip install -r requirements.txt
```

## Usage

1. **Run the Application**

```
python main.py
```
2. **Accept model downloads (only on the first run, roughly 6GB)**

3. **Follow the instructions provided in CLI** to:

    - **Build or reuse existing FAISS index**
    - **Load PDF files** via paths (sample PDF files can be found in the data folder, sourced from [CUAD_v1](https://zenodo.org/records/4595826))
    - **Ask questions** relevant to loaded PDF files

## Cleanup

After removing the repository files, don’t forget to also remove the downloaded models. By default, they are located in the `~/.cache/huggingface` folder.

## Project structure
```
legalbot
├── README.md
├── data 
│   └── raw_docs
│       ├── Part_I
│       ├── Part_II
│       └── Part_III
├── main.py
├── modules
│   ├── __init__.py
│   ├── utils.py
│   └── vector_store_builder.py
└── requirements.txt
```

## Notes

- The program will warn you about using a smaller context window (8192) compared to the context window used during training (32768). This is done due to memory constraints. If you intend to use a larger context window by modifying the constants in `main.py`, make sure you have enough RAM.
- Currently the models run on CPU.
- If you switch to a different LLM, you may need to adjust the model prompt.
- The program uses a FAISS database with Pickle. Using .pkl files requires `allow_dangerous_deserialization` flag in the `FAISS.load_local()` method to be set to `True`. This is perfectly safe if used as intended, as the .pkl files are created by the program after creating the FAISS index. Never use this program on .pkl files sourced from somewhere else, as they could potentially contain malicious code.

## Example usage

```
> python main.py
Embedding model 'intfloat/multilingual-e5-base' is already cached.
LLM file 'mistral-7b-instruct-v0.2.Q4_K_M.gguf' is already cached.

Loading models...
llama_context: n_ctx_per_seq (8192) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
Models loaded successfully.

No existing FAISS index found. Creating a new one.
Enter paths to PDF files (type 'done' when finished):
PDF file path: data/raw_docs/Part_I/Affiliate_Agreements/CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf
PDF file path: done
Indexed document paths saved to indexed_docs.json
FAISS index saved to: vector_store

Enter your question (type 'exit' to quit): What do I need to do to enroll in the Affiliate Program?
 To enroll in the Affiliate Program, you must submit a complete "Affiliate Registration Form" via the Chase Affiliate Website. 
 For new affiliates, use this link: <https://ssl.linksynergy.com/php-bin/reg/sregister.shtml?mid=2291>. 
 For existing affiliates, use this link: <http://www.linkshare.com/joinprograms?oid=87909>. 
 Chase will evaluate your registration form and notify you via e-mail of the acceptance or rejection.

Enter your question (type 'exit' to quit): What credit card descriptions can I use?
 You can only use credit card descriptions provided or approved in writing by Chase.

Enter your question (type 'exit' to quit): exit
Exiting the legal assistant. Goodbye!
```


## Roadmap

- Add GPU support for faster model inference
- Implement contextual splitting of PDFs into meaningful chunks, leveraging metadata to improve similarity search results
- Refine LLM prompting strategies for more accurate and relevant responses
- Develop automated correctness tests using a "Judge" LLM or embedding similarity metrics
- Enhance error handling and add tamper-proofing mechanisms for safer usage
- Build a web-based GUI for improved accessibility and user experience
- Additional features (support for other formats, specific document scope...)
