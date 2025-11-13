from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from dotenv import load_dotenv


# Load environment variables (gemini api key) from .env file
load_dotenv()
app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://aiplanet-frontend.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Global variable to store PDF text across functions
global pdfText, vector_store
pdfText = ""
vector_store = None  


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Load FAISS vector store if available, else set to None
try:
    vector_store = FAISS.load_local(
        "aiplanet_index", embeddings, allow_dangerous_deserialization=True
    )
except Exception as e:
    vector_store = None
    print(f"Failed to load FAISS index at startup: {e}")

# Function to read and extract text from a PDF
def get_pdf_txt(pdf_doc):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text


# Function to split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save vector store from text chunks

def get_vector_store(text_chunks):
    global vector_store  # Ensure we modify the global variable
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("aiplanet_index")


# Function to create prompt with a specific question and context
def get_prompt(context, question):
    prompt_template = f"""
    Answer the question as detailed as possible from the provided context,make sure to provide all the details in 100 words not more than that, if the answer is not in the
    provided context just say," answer is not available in the context",dont provide the wrong answer but if the question is related to provided content u can give 2 lines but dont provide more\n\n
    Context:\n {context}>\n
    Question: \n{question}\n
    
    Answer:
    """
    return prompt_template

# Basic endpoint to confirm the API is running
@app.get("/")
def index():
    global pdfText
    pdfText = ""
    return "successful"

# Endpoint to upload a PDF file and extract its text
@app.post("/")
async def upload(files: UploadFile = File(...)):
    global pdfText, vector_store
    allText = ""
    pdf = await files.read()
    pdf_doc = BytesIO(pdf)
    text = get_pdf_txt(pdf_doc)
    allText = text
    pdfText += allText

    # Create vector store if it doesn't exist
    if not vector_store:
        text_chunks = get_text_chunks(pdfText)
        get_vector_store(text_chunks)
        if vector_store is None:
            return "Failed to create vector store"
    return files.filename

# Endpoint to handle message sending with a question to the AI
@app.post("/sendmessage")
async def sendMessage(question: str):
    global pdfText, vector_store
    try:
        if not vector_store:
            return "Please upload a document first."
        context = vector_store.similarity_search(question, k=5)
        prompt = get_prompt(context, question)
        result = llm.invoke(prompt)
        return result.content

    except Exception as e:
        print(f"Unable to process: {e}")
        return {"error": f"An error occurred: {e}"}
