import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch

# Page setup
st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫", layout="wide")

# Categories
CATEGORIES = [
    "Connectivity Issue",
    "Hardware Malfunction",
    "Data Recovery",
    "Battery Issue",
    "Account Access",
    "Performance Issue",
    "Software Issue"
]

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("dataset/support_ticket_data.csv")

# Load fine-tuned model
@st.cache_resource
def load_fine_tuned_model():
    import os
    from transformers import AutoConfig
    
    # Try to find the model in different possible locations
    possible_paths = [
        "./fine_tuned_model",
        "fine_tuned_model",
        "/mount/src/auto-tagging-support-tickets-using-llm/fine_tuned_model"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            model_path = path
            break
            
    if model_path is None:
        st.error("❌ Could not find the fine-tuned model directory. Please ensure the 'fine_tuned_model' folder exists in the project root.")
        st.stop()
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load config
        config = AutoConfig.from_pretrained(model_path)
        
        # Initialize model with config
        model = AutoModelForSequenceClassification.from_config(config)
        
        # Try to load weights from safetensors if available
        safetensors_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
            # Remove any unexpected keys that might cause issues
            model_state_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
            model_state_dict.update(filtered_state_dict)
            model.load_state_dict(model_state_dict, strict=False)
        
        return tokenizer, model
            
    except Exception as e:
        # Provide more detailed error information
        import traceback
        error_details = traceback.format_exc()
        st.error(f"❌ Error loading the model: {str(e)}\n\nDebug info:\n{error_details}")
        st.stop()

# Load FLAN-T5 model for few-shot learning
@st.cache_resource
def load_flan_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Fine-tuned classification
def classify_fine_tuned(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    top3_idx = torch.topk(probs, 3).indices.tolist()
    top3_labels = [CATEGORIES[i] for i in top3_idx]
    top3_probs = [f"{probs[i].item()*100:.1f}%" for i in top3_idx]
    return list(zip(top3_labels, top3_probs))

# Few-shot classification with FLAN-T5
def get_top3_tags(text, tokenizer, model):
    categories = CATEGORIES
    prompt = f"""Classify this support ticket into the most relevant category. 
    Choose from: {', '.join(categories)}.
    
    Example 1: "My internet is not working"
    Category: Connectivity Issue
    
    Example 2: "My laptop won't turn on"
    Category: Hardware Malfunction
    
    Example 3: "I accidentally deleted important files"
    Category: Data Recovery
    
    Now classify this ticket: "{text}"
    Category:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=20)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Ensure prediction is a valid category, fallback to first category if invalid
    if prediction not in categories:
        prediction = categories[0]
    
    top3 = [
        (prediction, "High"),
        (categories[(categories.index(prediction) + 1) % len(categories)], "Medium"),
        (categories[(categories.index(prediction) + 2) % len(categories)], "Low")
    ]
    return top3

# Main app
def main():
    st.title("🎫 Support Ticket Classifier")

    # Load resources
    data = load_data()
    tokenizer, model = load_fine_tuned_model()
    flan_tokenizer, flan_model = load_flan_model()

    # Initialize session state for ticket text
    if "ticket_text" not in st.session_state:
        st.session_state.ticket_text = ""

    # Model selection
    st.subheader("Classify a New Ticket")
    model_choice = st.radio("Choose classification method:", ["Fine-tuned Model", "FLAN-T5 Few-shot"], key="model_choice")

    # Input for new ticket
    ticket_text = st.text_area(
        "Enter the support ticket text:",
        height=150,
        placeholder="Type or paste the support ticket here...",
        value=st.session_state.ticket_text,  # Use session state to populate text area
        key="ticket_input"
    )

    # Update session state when text area changes
    if ticket_text != st.session_state.ticket_text:
        st.session_state.ticket_text = ticket_text

    if st.button("Classify Ticket"):
        if ticket_text.strip():
            with st.spinner("Analyzing ticket..."):
                results = (
                    classify_fine_tuned(ticket_text, tokenizer, model)
                    if model_choice == "Fine-tuned Model"
                    else get_top3_tags(ticket_text, flan_tokenizer, flan_model)
                )
                st.subheader("Top 3 Predicted Categories:")
                for i, (label, prob) in enumerate(results, 1):
                    st.success(f"{i}. {label} ({prob})")
        else:
            st.warning("Please enter some text to classify.")

    # Sample ticket section
    st.subheader("Sample Tickets from Dataset")
    if st.checkbox("Show sample tickets"):
        sample = data.sample(5, random_state=42)  # Fixed random_state for consistency
        for _, row in sample.iterrows():
            with st.expander(f"Ticket {row['support_tick_id']}"):
                st.write(row["support_ticket_text"])
                if st.button("Load this sample", key=f"load_{row['support_tick_id']}"):
                    st.session_state.ticket_text = row['support_ticket_text']
                    st.rerun()

if __name__ == "__main__":
    main()