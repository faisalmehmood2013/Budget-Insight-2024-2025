# Used for building interactive web applications and dashboards. 
import streamlit as st
from PIL import Image

# Used for reading and manipulating PDF files. 
from PyPDF2 import PdfReader

# Provides text splitting functionality for preprocessing text data.
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# Includes embeddings models like OllamaEmbeddings and HuggingFaceEmbeddings, which can be used for encoding text into numerical vectors.
# from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Supports various vector storage systems like Cassandra, Chroma, and FAISS for efficient storage and retrieval of embeddings.
# from langchain.vectorstores import Cassandra, Chroma, FAISS
from langchain_community.vectorstores import Cassandra

# Provides a wrapper for integrating vector stores with LangChain.
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

# Includes functionality for creating retrieval and combination chains for working with documents.
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Provides access to chat models like ChatOpenAI.
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

# Includes chat prompt templates
from langchain_core.prompts import ChatPromptTemplate

# Used for interacting with the operating system
import os

# A Python client for the Cassandra database
import cassio

# Allows loading environment variables from a .env file.
from dotenv import load_dotenv

# Used for timing the execution of code
import time

# Loads environment variables from a .env file.
load_dotenv()

# Set the environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
openai_base_url = os.getenv('OPENAI_URL')

# Connection of the ASTRA DB
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

# Initialize the Cassandra client
cassio.init(database_id=ASTRA_DB_ID, token=ASTRA_DB_APPLICATION_TOKEN)

# Initialize the OpenAI model
llm = ChatOpenAI(openai_api_base=openai_base_url, temperature=0.5)

# Initialize the embeddings model
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>

Question: {input}""")

# Streamlit UI
st.set_page_config(page_title="Budget Insight | 2024-2025")
st.header("Budget Insight | 2024-2025")

def vector_embedding():
    if "vectors" not in st.session_state:

        # Creating a PdfReader Object
        st.session_state.pdfreader = PdfReader("Budget_Insight_2024-25.pdf")                

        # read text from PDF
        st.session_state.raw_text = ""
        for i, page in enumerate(st.session_state.pdfreader.pages):
            content = page.extract_text()
            if content:
                st.session_state.raw_text += content

        # Splitting the text
        st.session_state.text_splitter = CharacterTextSplitter(chunk_size = 800, chunk_overlap = 80, length_function = len, separator = "\n")
        st.session_state.texts= st.session_state.text_splitter.split_text(st.session_state.raw_text)

        # Embedding the text
        st.session_state.embeddings = HuggingFaceEmbeddings()

        # Initialize the vector store
        st.session_state.astra_vector_store = Cassandra(
            embedding=st.session_state.embeddings,
            table_name="text_chunk_demo",
            session=None,
            keyspace=None,
        )

        # Add the texts to the vector store
        st.session_state.astra_vector_store.add_texts(st.session_state.texts)

        # Create the index
        st.session_state.astra_vector_index = VectorStoreIndexWrapper(vectorstore=st.session_state.astra_vector_store)

prompt1=st.text_input("Enter Your Question From Budget Insight | 2024-2025")

if st.button("Search"):
    vector_embedding()
    st.write("Answer Is Ready.......")

if prompt1:

    # Creating a Vector Store Retriever
    retriever=st.session_state.astra_vector_store.as_retriever()

    # Create the chain
    document_chain=create_stuff_documents_chain(llm,prompt)

    # Create the retrieval chain
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

st.header("Information")
st.write('''Welcome, user!

Thank you for accessing the "Budget Insight 2024-25" document provided by RSM Pakistan. This comprehensive publication offers valuable insights into the economic outlook and the key changes proposed in the Finance Bill 2024-25 that will impact businesses and individuals in Pakistan.

The document covers a range of important topics, including:

The global economic outlook and its potential implications for Pakistan
An analysis of Pakistan's economic performance, with details on GDP growth, per capita income, investment-to-GDP ratio, and inflation trends
Proposed amendments to the Income Tax Ordinance, Sales Tax Act, Federal Excise Act, and Customs Act
Changes in other relevant laws and regulations
We hope that this guide will serve as a useful reference point as you navigate the upcoming 2024-25 fiscal year in Pakistan. Please feel free to reach out to our partners and experts listed on the back cover if you have any further questions or require additional information.

We appreciate your interest and trust that you will find this publication informative and insightful.''')

st.sidebar.markdown("### Connect with us on social media:")
st.sidebar.markdown("""
- [Facebook](https://www.facebook.com/FMGillani01)
- [Twitter](https://twitter.com/FMGillani)
- [LinkedIn](https://www.linkedin.com/in/fmehmood1122/)
- [Instagram](https://www.instagram.com/fmgillani/)
""")

# Create the sidebar
st.sidebar.title("Some Sample Questions on Budget Insight 2024-25")
questions = [
    "1. What is Current GDP?",
    "2. How many million tonnes of rice were produced in FY 2023-24?",
    "3. What percentage of GDP is attributed to livestock?",
    "4. what were the key factors contributing to the growth of the agriculture sector in FY24, and how did this impact overall economic performance?",
    "5. How did the value of the Pakistan rupee change against the US dollar during the first eleven months of FY24?",
    "6. How many inflow increase Foreign Direct Investment in July-Apr?",
    "7. How did the fiscal deficit as a percentage of GDP in FY24 compare to the same period last year?",
    "8. How much increase in tax on vehicle registration?",
    "9. What was the contribution of the services sector to Pakistan's GDP in the current fiscal year (2024-25)?",
    "10. What is the forecasted growth of the global AI market by 2025?",
    "11. What is the projected GDP growth rate for China in 2024?"
    "12. What is the overall economic growth outlook for Pakistan in the 2024-2025 fiscal year?",
    "13. How has the performance of the major sectors of the Pakistani economy, such as agriculture, manufacturing, and services, been in the current fiscal year compared to the previous year?",
    "14. What are the key factors that have affected the investment-to-GDP ratio in Pakistan during the 2024 fiscal year?",
    "15. What are the notable developments in the global economy and their potential impact on Pakistan's economic outlook?",
    "16. How has the inflation rate in Pakistan, as measured by the Consumer Price Index (CPI), trended in the current fiscal year compared to the previous year?",
    "17. What are the key growth figures and trends for important crops like wheat, cotton, and rice in the agriculture sector of Pakistan?",
    "18. How has the performance of the large-scale manufacturing sector in Pakistan been during the current fiscal year compared to the previous year?",
    "19. What are the proposed changes or amendments related to direct taxes (Income Tax Ordinance) in the 2024-25 budget?",
    "20. What are the key proposals related to indirect taxes such as sales tax, federal excise duty, and customs duty in the 2024-25 budget?",
    "21. What are the notable changes or amendments in the Customs Act, 1969 proposed in the 2024-25 budget?",
    "22. How does the budget proposal address the issue of tax evasion and avoidance in Pakistan?",
    "23. What are the key incentives or concessions offered to specific industries or sectors in the 2024-25 budget?",
    "24. What are the proposed changes in the taxation of dividends and capital gains in the 2024-25 budget?",
    "25. How does the budget aim to promote investment and economic growth in the country?",
    "26. What are the measures introduced to simplify the tax compliance and administration process for businesses and individuals?",
    "27. How does the budget address the issue of revenue generation and fiscal deficit management?",
    "28. What are the potential implications of the proposed changes in the Sales Tax Act, 1990 on businesses and consumers?",
    "29. What are the key amendments in the Federal Excise Act, 2005 that will impact various industries?",
    "30. How does the budget seek to encourage the use of technology and digitalization in tax administration?",
    "31. What are the other notable changes or reforms introduced in the 2024-25 budget beyond the tax-related provisions?"
]

# Display the questions in the sidebar
for question in questions:
    st.sidebar.write(question)




