import spacy

nlp = spacy.load("en_core_web_sm")
text = "I love my new Samsung Galaxy phone. Beats my old Nokia by far!"

doc = nlp(text)

# Named Entity Recognition
for ent in doc.ents:
    print(ent.text, ent.label_)

# Rule-based sentiment (simple demo)
positive_words = ["love", "great", "amazing", "beats"]
negative_words = ["bad", "hate", "poor"]

score = sum([1 if token.text.lower() in positive_words else -1 if token.text.lower() in negative_words else 0 for token in doc])
sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
print("Sentiment:", sentiment)
