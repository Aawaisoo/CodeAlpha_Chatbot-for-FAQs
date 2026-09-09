
# FAQ Chatbot

A simple FAQ chatbot built with Streamlit and TF-IDF text similarity. It matches a user's question to the closest predefined FAQ and returns the corresponding answer.

## How It Works

1. A set of question-answer pairs (FAQs) is stored in a dictionary.
2. Each question is preprocessed (lowercased, tokenized, stopwords removed).
3. Questions are converted into TF-IDF vectors.
4. When a user types a question, it's preprocessed and compared against all FAQ vectors using cosine similarity.
5. The closest matching FAQ's answer is returned. If the similarity score is below 0.3, the bot responds with a fallback message.

## Requirements

- Python 3.7+
- streamlit
- nltk
- scikit-learn

Install dependencies:

```bash
pip install streamlit nltk scikit-learn
```

## Running the App

```bash
streamlit run chatbot.py
```

This opens the chatbot in your browser. Type a question in the input box and click **Send** to get a response. Click **See available FAQs** to view all supported questions.

## Project Structure

```
chatbot.py   # Main application file
```

## Notes

- NLTK resources (`punkt`, `punkt_tab`, `stopwords`) are downloaded automatically on first run.
- The FAQ list can be extended by adding more entries to the `faqs` dictionary in `chatbot.py`.
