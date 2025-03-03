from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter # Keep import but make chunking optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from io import BytesIO

# Set Google API key
os.environ["GOOGLE_API_KEY"] = 'Add your gemini api key'
google_api_key = os.getenv("GOOGLE_API_KEY")

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    if isinstance(pdf_file, BytesIO):  # If already a file-like object
        pdfReader = PdfReader(pdf_file)
    else:  # If it's a file path
        pdfReader = PdfReader(pdf_file)

    all_text = ""
    for page in pdfReader.pages:
        text = page.extract_text()
        if text:
            all_text += text.encode('ascii', 'ignore').decode('ascii') + "\n"
    return all_text

# # Function to extract text from a webpage # Removed as per request
# def extract_text_from_url(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     article_content = soup.find_all('p')
#     text = '\n'.join([p.get_text() for p in article_content])
#     return text.encode('ascii', 'ignore').decode('ascii')

# Function to split text into smaller chunks (now optional - set large chunk size to effectively disable)
def get_text_chunks(text, chunking_enabled=True): # Added chunking_enabled flag
    if chunking_enabled:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) # Large chunk size to minimize splitting
        return text_splitter.split_text(text)
    else:
        return [text] # If chunking disabled, treat the whole text as a single chunk


# Function to store documents permanently in FAISS
def update_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")

    # Load existing FAISS DB if available, otherwise create a new one
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        vector_store.add_texts(text_chunks)  # Add new data to existing FAISS index
    else:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    vector_store.save_local("faiss_index")  # Save updated FAISS index

#Function to load documents into FAISS permanently
def load_documents(chunk_document=True): # Added chunk_document parameter to control chunking
    all_text = ""

    # List of PDF files to add permanently
    pdf_files = [r""] # Replace with your actual PDF path if needed, or keep it as is if the path is correct
    print(f"Loading PDF files: {pdf_files}") # DEBUGGING STEP 1: Check if load_documents is called and PDF list

    for pdf in pdf_files:
        print(f"Processing PDF: {pdf}") # DEBUGGING STEP 2: Check if loop starts and PDF path is printed
        try:  # Added try-except to catch file opening errors
            with open(pdf, "rb") as file:
                pdf_text = extract_text_from_pdf(file)
                print(f"Text extracted from PDF, length: {len(pdf_text)}") # DEBUGGING STEP 3: Check if text is extracted and length
                all_text += pdf_text
        except Exception as e:
            print(f"Error opening or processing PDF {pdf}: {e}") # DEBUGGING STEP 4: Catch PDF errors

    # # List of URLs to add permanently # Removed URL loading
    # urls = []
    # for url in urls:  # No changes needed for URLs if you're not using them currently
    #     all_text += extract_text_from_url(url)

    # Process and store documents in FAISS
    if chunk_document:
        text_chunks = get_text_chunks(all_text, chunking_enabled=True) # Chunking enabled by default
        print(f"Text chunking ENABLED. Number of text chunks created: {len(text_chunks)}") # DEBUGGING STEP 5a: Chunking enabled message
    else:
        text_chunks = get_text_chunks(all_text, chunking_enabled=False) # Chunking disabled
        print(f"Text chunking DISABLED. Treating document as single chunk.") # DEBUGGING STEP 5b: Chunking disabled message


    update_vector_store(text_chunks)
    print("FAISS index updated.") # DEBUGGING STEP 6: Confirm FAISS update


# Compliance check function
def get_compliance_chain():
    prompt_template = """
    You are an AI compliance model for the 2 documents provided in the context which are sebi s' dcoument , your role is to analyse the user s' work in detail based on the regulation document provided in the context.
    Follow these steps:
    1. Please ignore the place holders in the document they are for later use and are not errors
    2. Focus primarily on factual inconsistencies, numerical mismatches, and compliance errors, even if the text appears grammatically correct. Ignore minor typographical and formatting issues unless they impact meaning. 
    3. Check if ICDR regulation 6 is present 
    
    Context (Document):\n {context} \n
    User Submission:\n {submission} \n

    Document Assessment:
    """
    # Removed the strict "missing context" condition.
    # Changed role to "AI document analyst"
    # Changed focus to "document information," "guidelines or recommendations," "assessment," "alignment/deviation/consideration"

    model = ChatGoogleGenerativeAI(google_api_key=google_api_key, model="gemini-2.0-flash-thinking-exp-01-21", temperature=0.1, max_output_tokens=10000)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "submission"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to check compliance against stored regulatory documents
def check_compliance(user_submission):
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    print(f"User Submission for Similarity Search: '{user_submission}'") # DEBUGGING STEP 7: Print user submission
    # Retrieve relevant regulatory content for evaluation
    relevant_docs = vector_store.similarity_search(user_submission, k=25)
    print(f"Number of relevant documents retrieved: {len(relevant_docs)}") # DEBUGGING STEP 8: Check number of retrieved docs

    if not relevant_docs: # DEBUGGING STEP 9: Check if no docs are retrieved
        print("No relevant documents found by similarity search!")
    else:
        print("First retrieved document (for debugging):") # DEBUGGING STEP 10: Print content of first retrieved doc
        print(f"Page Content (first 200 chars): {relevant_docs[0].page_content[:200]}...")


    # Run compliance check
    chain = get_compliance_chain()
    response = chain({"input_documents": relevant_docs, "submission": user_submission}, return_only_outputs=True)

    return response, relevant_docs

# Run this once to store compliance documents permanently (Only run this once initially, or when you update documents)
# load_documents()  # Commented out after initial loading. Uncomment to reload documents if needed.


# Function to run the chatbot
def run_chatbot():
    print("Welcome to the Compliance Chatbot! ")
    print("I am ready to check your submissions against the loaded regulatory documents.")
    print("Type 'exit' or 'quit' to end the chat.")

    while True:
        user_submission = input("\nYour Submission: ")

        if user_submission.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Thank you!")
            break

        if not user_submission.strip():  # Handle empty input
            print("Please enter a submission to check for compliance.")
            continue

        try:
            response, relevant_docs = check_compliance(user_submission)
            compliance_assessment = response['output_text']

            print("\nDocument Assessment:") # Changed print statement to reflect relaxed prompt
            print(compliance_assessment)

            # Optional: Print relevant documents for debugging or context
            # print("\nRelevant Documents (for context):")
            # for doc in relevant_docs:
            #     print(f"Page Content: {doc.page_content[:200]}...") # Print first 200 chars of each doc

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again or check if the FAISS index is properly loaded and the API key is set correctly.")


if __name__ == "__main__":
    # Control chunking here: chunk_document=True (chunked), chunk_document=False (unchunked/whole document)
    use_chunking = True # Set to False to disable chunking and load whole document

    print(f"Document chunking is {'ENABLED' if use_chunking else 'DISABLED'} for this run.") # Indicate chunking status

    # Force loading documents and creating FAISS index every time for debugging (you can remove this later if needed)
    print("Forcing document loading and FAISS index creation on each run for debugging...") # Added message
    load_documents(chunk_document=use_chunking) # Pass chunk_document parameter to load_documents
    print("Documents loaded and FAISS index (re-)created.") # Confirmation message

    run_chatbot()
