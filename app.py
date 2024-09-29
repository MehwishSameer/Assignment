import streamlit as st
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Title of the web application
st.title("OCR and Document Search Web Application Prototype")

# Initialize session state for storing extracted text
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

# File uploader for image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Process the uploaded image only if a new file is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Load the model (cached so it doesn't reload each time)
    @st.cache_resource
    def load_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
        return model, processor, device

    model, processor, device = load_model()

    # Define text query
    text_query = "Extract both the hindi and english text. Ignore bounding boxes. Do not return any coordinates, only return plain text."

    # Generate inputs and outputs
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text_query}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=6144)

    # Decode the generated output
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Convert '\n' to spaces and store the result in session state
    st.session_state.extracted_text = ' '.join(output_text).replace('\n', ' ')
    st.write("Extracted Text:", st.session_state.extracted_text)

# Keyword search functionality
keyword = st.text_input("Enter a keyword to search within the text")
if keyword:
    # Make keyword search case-insensitive
    if keyword.lower() in st.session_state.extracted_text.lower():
        st.write(f"Keyword '{keyword}' found in the text.")
    else:
        st.write(f"Keyword '{keyword}' not found in the text.")
