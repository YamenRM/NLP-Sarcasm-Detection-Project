import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


model_name = "YamenRM/sarcasm_model"

# load pre-trained model and tokenizer
@st.cache_resource
def load_model():
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name)

  return tokenizer, model

tokenizer, model = load_model()


st.title("Sarcasm Detection App")
st.write("Enter a sentence to determine if it is sarcastic or not.")

# input text area
user_input = st.text_area( "Type your text here...")

# button to trigger prediction
if st.button("detect sarcasm"):
    if user_input.strip() == "":
        st.write("Please enter a valid sentence.")
    else:
        # tokenize input text
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        # get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            proba = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(proba, dim=-1).item()
            confidence = proba[0][prediction].item()
        
        # display result
        label= "Sarcastic 🤨" if prediction == 1 else "Not Sarcastic 🙂"
        st.subheader(label)
        st.write(f"Confidence: {confidence:.2f}")
                 


