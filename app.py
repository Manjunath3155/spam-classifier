import gradio as gr
import joblib

# Load model
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def classify_spam(message):
    input_vectorized = vectorizer.transform([message])
    prediction = model.predict(input_vectorized)[0]
    return "🚨 Spam!" if prediction == 1 else "✅ Not Spam."

# Create UI
iface = gr.Interface(
    fn=classify_spam,
    inputs=gr.Textbox(placeholder="Enter your message..."),
    outputs=gr.Label(),
    title="📩 Spam Detector",
    description="Type a message below to check if it's spam or not."
)
iface.launch()

