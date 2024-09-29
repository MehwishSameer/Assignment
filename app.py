import streamlit as st
from PIL import Image
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Title of the web application
st.title("Hindi-English Text Extractor")

# Load models once and store them in session state
if 'model' not in st.session_state:
    st.session_state.model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    st.session_state.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)

# File uploader for image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Define text query
    text_query = "Extract both the hindi and english text. Ignore bounding boxes. Do not return any coordinates, only return plain text."

    # Prepare messages for the model
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text_query}]}]
    text = st.session_state.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    # Create inputs for the model
    inputs = st.session_state.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    inputs = {key: value.to('cuda' if torch.cuda.is_available() else 'cpu') for key, value in inputs.items()}

    # Generate output using the model
    generated_ids = st.session_state.model.generate(**inputs, max_new_tokens=1024)  # Reduce max tokens if feasible

    # Decode the generated output
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
    output_text = st.session_state.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Convert '\n' to spaces
    formatted_text = ' '.join(output_text).replace('\n', ' ')
    
    st.write("Extracted Text:", formatted_text)

    # Keyword search functionality
    keyword = st.text_input("Enter a keyword to search within the text")
    if keyword:
        if keyword in formatted_text:
            st.write(f"Keyword '{keyword}' found in the text.")
        else:
            st.write(f"Keyword '{keyword}' not found in the text.")
