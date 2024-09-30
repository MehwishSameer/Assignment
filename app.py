import gradio as gr
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import time
import re  # For keyword search and highlighting

# Load the model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor with caching
def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    return model, processor

model, processor = load_model()

# Extract text from the image using Qwen2-VL
def extract_text_from_image(image):
    if image is None:
        return "No image uploaded", ""

    start_time = time.time()

    # Define the text query for OCR
    text_query = ("Extract both the hindi and english entire texts. Hindi should be extracted in hindi and english in english. No need to explicitly tag it as english or hindi. Ignore bounding boxes. "
                  "Do not return any coordinates, only return plain text.")

    # Prepare input messages
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text_query}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate text using the model
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Convert '\n' to spaces and join the output
    extracted_text = ' '.join(output_text).replace('\n', ' ')

    end_time = time.time()
    processing_time = f"OCR processing time: {end_time - start_time:.2f} seconds"
    
    return extracted_text, processing_time

# Keyword search in the extracted text and highlight
def keyword_search(text, keyword):
    # Remove any previous highlights and not found messages
    def clean_text(text):
        # Remove <mark> tags and any "Keyword not found" messages
        cleaned_text = re.sub(r"<mark>(.*?)</mark>", r"\1", text)
        return re.sub(r"Keyword '.*?' not found\.", "", cleaned_text).strip()

    # Highlight the keyword in the text
    def highlight_keywords(text, keyword):
        highlighted_text = re.sub(f"({re.escape(keyword)})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
        return highlighted_text

    # Clean previous highlights and messages
    text = clean_text(text)

    if keyword:
        if keyword.lower() in text.lower():
            highlighted_text = highlight_keywords(text, keyword)
            return highlighted_text
        else:
            return f"{text}<br>Keyword '{keyword}' not found."
    else:
        return text  # If no keyword is provided, return the plain text

# Gradio interface
def image_processing(image):
    extracted_text, processing_time = extract_text_from_image(image)
    return extracted_text, processing_time

def search_in_text(extracted_text, keyword):
    # Update the extracted text with new highlights or no highlights
    highlighted_text = keyword_search(extracted_text, keyword)
    return highlighted_text

# Define Gradio app layout
with gr.Blocks() as app:
    gr.Markdown("# OCR and Document Search Web Application Prototype")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            
        with gr.Column():
            keyword_input = gr.Textbox(label="Enter a keyword to search")
            extracted_text_output = gr.HTML(label="Extracted Text")
            processing_time_output = gr.Textbox(label="OCR Processing Time", interactive=False)

    # Run image processing when image is uploaded
    image_input.change(fn=image_processing, inputs=image_input, outputs=[extracted_text_output, processing_time_output])

    # Run keyword search and update extracted text with highlighted keyword
    keyword_input.submit(fn=search_in_text, inputs=[extracted_text_output, keyword_input], outputs=extracted_text_output)

# Launch the Gradio app
app.launch()
