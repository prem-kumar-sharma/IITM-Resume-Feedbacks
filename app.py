from flask import Flask, request, jsonify, render_template
import PyPDF2
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI  # Updated import for OpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import logging

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the environment variables from the .env file
load_dotenv()

# Initialize Flask App
app = Flask(__name__)

# Load the OpenAI API Key securely from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure the API key is loaded, or raise an error
if not openai_api_key:
    raise EnvironmentError("OpenAI API key not found. Ensure it's set in the environment variables.")

# Load Resume Guidelines from PDF file
guideline_file_path = "Resume Guidelines for IIT Madras BS Degree - Aug 2024.pdf"

# Check if the guideline file exists before loading
if not os.path.exists(guideline_file_path):
    raise FileNotFoundError(f"Guideline file not found at {guideline_file_path}")

# Process the PDF file
guideline_text_list = []
with open(guideline_file_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        guideline_text_list.append(page_text)

# Combine all extracted text from the guidelines into a single string
guideline_text = "\n".join(guideline_text_list)

# Split and chunk the guideline text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
guideline_chunks = text_splitter.split_text(guideline_text)

# Custom embedding class using SentenceTransformer
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return [embedding.tolist() for embedding in self.model.encode(texts, show_progress_bar=True)]

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=True)[0].tolist()

# Initialize the SentenceTransformer model
embedding_model = SentenceTransformerEmbeddings('sentence-transformers/all-MiniLM-L6-v2')

# Convert the guideline chunks to embeddings and store them in a Chroma vector database
guideline_vector_db = Chroma.from_texts(
    texts=guideline_chunks,
    embedding=embedding_model,
    collection_name="resume-guidelines"
)

# Create a retriever from the guideline vector database
guideline_retriever = guideline_vector_db.as_retriever()

# Define the system prompt template for generating feedback
feedback_system_prompt_template = """
System Prompt:

You are a Resume Review Expert trained using a resume guideline document. Your task is to provide detailed feedback on the resume provided by the user. Make sure your feedback is based entirely on the resume guidelines document, and avoid introducing any assumptions.

Provide actionable feedback in the following areas:
1. Overall Structure
2. Formatting and Clarity
3. Relevance of Information
4. Any additional suggestions for improvement

Make sure your feedback is concise and follows the structure mentioned above.

Context (Resume): {context}
Resume Guidelines: {guidelines}
"""

# Create a prompt template for generating resume feedback
feedback_prompt = PromptTemplate(
    input_variables=["context", "guidelines"],
    template=feedback_system_prompt_template
)

# Initialize the OpenAI ChatGPT model (updated with correct ChatOpenAI import)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

# Function to generate feedback based on the resume and guidelines
def generate_resume_feedback(resume_text):
    try:
        # Retrieve the most relevant context from the guidelines
        retrieved_guideline_docs = guideline_retriever.invoke(resume_text)  # Updated method invoke()
        guideline_context = "\n".join([doc.page_content for doc in retrieved_guideline_docs])

        # If enough guideline context is found, generate feedback
        if guideline_context.strip():
            # Format the prompt with the resume text and guideline context
            prompt = feedback_prompt.format(context=resume_text, guidelines=guideline_context)
            response = llm.invoke(prompt)  # Use invoke instead of __call__

            # Extract the message content (assuming it's a string)
            feedback = response.content  # Use the correct method to extract the content

            return feedback
        else:
            return "No relevant guideline context found to provide feedback."
    except Exception as e:
        logging.error(f"Error generating feedback: {e}")
        return "An error occurred while generating feedback. Please try again later."



@app.route('/')
def index():
    logging.info("Rendering home page")
    return render_template('index.html')

@app.route('/evaluator')
def evaluator():
    logging.info("Rendering evaluator page")
    return render_template('evaluator.html')

# Route for processing the resume
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        
        # Process the uploaded resume (PDF only for now)
        resume_text_list = []
        if file.filename.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                resume_text_list.append(page_text)
        
        # Combine the text from the resume
        resume_text = "\n".join(resume_text_list)

        # Generate feedback
        feedback = generate_resume_feedback(resume_text)
        
        return jsonify({"feedback": feedback})
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        return jsonify({"error": "Something went wrong. Please try again later."}), 500

# Run the Flask app
if __name__ == '__main__':
    try:
        logging.info("Starting Flask app...")
        app.run(debug=True, port=5001)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
