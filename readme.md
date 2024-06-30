# Budget Insight 2024-2025 Analysis

This project uses Streamlit, LangChain, and other libraries to analyze the "Budget Insight 2024-25" document. It provides a user-friendly interface to explore the economic outlook and key legislative changes proposed in Pakistan's Finance Bill for 2024-25.

## Features

- **Interactive Web Application**: Built with Streamlit for easy navigation and data exploration.
- **PDF Processing**: Utilizes PyPDF2 to read and extract text from the budget document.
- **Text Embedding**: Employs HuggingFaceEmbeddings for encoding text into numerical vectors.
- **Vector Storage**: Stores and retrieves embeddings efficiently using Cassandra.
- **AI-Driven Responses**: Integrates OpenAI models to generate intelligent responses based on user queries.
- **Prompt Templates**: Uses LangChain's ChatPromptTemplate for context-driven question answering.

## Install the required packages:

```bash
pip install -r requirements.txt
```

## Set up environment variables by creating a .env file with the necessary keys:

# Usage

1. Run the Streamlit application:
   ```python
   streamlit run app.py
   ```
2. Enter questions related to the budget in the input box and get detailed responses.

## **Sample Questions**
1. What is the current GDP?
2. How has the Pakistan rupee performed against the US dollar in FY24?
3. What are the proposed changes to the Income Tax Ordinance?

Connect with Us:
-   [LinkedIn](https://www.linkedin.com/in/fmehmood1122/)
-   [Facebook](https://www.facebook.com/FMGillani01)
-   [Twitter](https://twitter.com/FMGillani)
- [Instagram](https://www.instagram.com/fmgillani/)
