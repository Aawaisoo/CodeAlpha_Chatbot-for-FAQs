
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

# Step 1: Collect FAQs (question - answer pairs)
faqs = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hello! How can I assist you today?",
    "who made you?": "I was created by a team of developers to assist with FAQs.",
    "What is your name?": "I am a simple FAQ chatbot.",
    "What are your working hours?": "We are open from 9 AM to 5 PM, Monday to Friday.",
    "How can I contact support?": "You can email us at support@example.com.",
    "Where are you located?": "Our office is located in Islamabad, Pakistan.",
    "What is your refund policy?": "Refunds are processed within 7 business days.",
    "Do you offer online services?": "Yes, all our services are available online.",
}

questions = list(faqs.keys())
answers = list(faqs.values())


# Step 2: Preprocess text
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


processed_questions = [preprocess(q) for q in questions]

# Step 3: TF-IDF vectors
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)


# Step 4: Match user question with closest FAQ
def get_best_answer(user_question):
    processed_input = preprocess(user_question)
    input_vector = vectorizer.transform([processed_input])
    similarity_scores = cosine_similarity(input_vector, question_vectors)
    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match_index]

    if best_score < 0.3:
        return "Sorry, I don't have an answer for that. Please contact support."
    return answers[best_match_index]


# Step 5: Streamlit UI
st.title("FAQ Chatbot")
st.write("Ask me a question below.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", key="user_input")

if st.button("Send") and user_input:
    response = get_best_answer(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

for sender, message in st.session_state.chat_history:
    st.write(f"**{sender}:** {message}")

with st.expander("See available FAQs"):
    for q in questions:
        st.write("- " + q)
