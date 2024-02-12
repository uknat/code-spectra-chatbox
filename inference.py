def load_model_and_vectorizer(model_path, vectorizer_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

def predict(model, vectorizer, text):
    processed = preprocess_data(pd.DataFrame([{"text": text}]))[0]
    prediction = model.predict(processed)
    return prediction

model, vectorizer = load_model_and_vectorizer('chatbot_model.pkl', 'vectorizer.pkl')
print(predict(model, vectorizer, "Your input text here"))
