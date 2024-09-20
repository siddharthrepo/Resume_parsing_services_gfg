from flask import Flask, request , render_template , jsonify , session
import pickle ,re
from PyPDF2 import PdfReader
from utils import extract_contact_number_from_resume , extract_education_from_resume , extract_email_from_resume , extract_name_from_resume , extract_skills_from_resume
from utils import match


app = Flask(__name__)
app.secret_key = 'jajsjsnjdnadjdwejdwelfblefbew'

rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
    return predicted_category

# Prediction and Category Name
def job_recommendation(resume_text):
    resume_text= cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    recommended_job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return recommended_job

def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text



@app.route("/")
def main():
    return render_template('index.html')
@app.route("/pred" , methods=['POST'])
def pred():
    # Process the PDF or TXT file and make prediction
    if 'resume' in request.files:
        file = request.files['resume']
        filename = file.filename

        job_description = request.form['job_description']

        if filename.endswith('.pdf'):
            session['text'] = pdf_to_text(file)

        elif filename.endswith('.txt'):
            session['text'] = file.read().decode('utf-8')
        else:
            return render_template('resume.html', message="Invalid file format. Please upload a PDF or TXT file." )


        
        predicted_category = predict_category(session['text'])
        recommended_job = job_recommendation(session['text'])
        phone = extract_contact_number_from_resume(session['text'])
        email = extract_email_from_resume(session['text'])

        extracted_skills = extract_skills_from_resume(session['text'])
        extracted_education = extract_education_from_resume(session['text'])
        name = extract_name_from_resume(session['text'])
        score = match(job_description , session['text'])
        print(score)

        return render_template('index.html', predicted_category=predicted_category,recommended_job=recommended_job,
                               phone=phone,name=name,email=email,extracted_skills=extracted_skills,extracted_education=extracted_education , score = score)
    else:
        return render_template("index.html", message="No resume file uploaded.")


from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv("/home/siddharth/Desktop/resume_parser/.env")
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
model = ChatGoogleGenerativeAI(model="gemini-pro" , google_api_key = GOOGLE_API_KEY , temperature = 0.3 , convert_system_message_to_human = True)



@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    
    pages = session['text']

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000 , chunk_overlap = 1000)
    context = "".join(str(p) for p in pages)
    texts = text_splitter.split_text(context)
    
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001" , google_api_key=GOOGLE_API_KEY)
    
    vector_index = Chroma.from_texts(texts , embeddings).as_retriever(search_kwargs={"k" : 5})
    
    template = """
        Use the following pieces of context to answer the questions on the basis of the resume provided. . if you don't know the answer try to generate it . give short and clear answers . if any /n is their remove it .Always say "thanks for asking !" at the end of the answer
        {context}
        Question: {question}
        Helpful Answer:
    """
    qa_chain_Prompt =  PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        model ,
        retriever = vector_index,
        return_source_documents = True,
        chain_type_kwargs = {"prompt" : qa_chain_Prompt}
    )
    question = user_message
    result = qa_chain({"query" : question})
    response  = result["result"]
    # Simple chatbot logic
    return jsonify({"response": response})
   
    


if __name__ == "__main__":
    app.run(debug=True)