from PIL import Image
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Load and preprocess the image using OpenCV
image_path = "Image/image4.jpg"  # Replace with your image file path
image = Image.open(image_path)

# Initialize the RAG model
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")

# Index the image instead of PDF
RAG.index(
    input_path=image_path,  # Path to the image
    index_name="Image",
    store_collection_with_index=False,
    overwrite=True
)

# Load the Qwen-2VL model and move it to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device).eval()  # Move model to GPU and set to eval mode

# Define the text query you want to ask about the image
text_query = "Extract both the hindi and english text. Ignore bounding boxes. Do not return any coordinates, only return plain text."

# Search the RAG model for relevant information
results = RAG.search(text_query, k=1)
print("Search Results:", results)

# Initialize the processor to handle image-text interactions
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)

# Extract the index of the image (if there's pagination, use it; otherwise, assume single image)
image_index = 0  # Since there's just one image, we set the index to 0

# Prepare the messages in the format expected by the model
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,  # Use the loaded image here
            },
            {"type": "text", "text": text_query},
        ],
    }
]

# Generate the text processing input for the model
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Process the vision info (image inputs for the model)
image_inputs, video_inputs = process_vision_info(messages)


# Create inputs to the model
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Move inputs to GPU
inputs = {key: value.to(device) for key, value in inputs.items()}

input_length = inputs['input_ids'].shape[1]  # Get the length of the input

# Generate output using the Qwen-2VL model
generated_ids = model.generate(**inputs, max_new_tokens = 6144)

# Trim generated output and decode into human-readable text
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# Print the generated output text
print("Generated Output:", output_text)

output_text = ' '.join(output_text)

# Replace \n with a space
formatted_text = output_text.replace('\n', ' ')

# Print the formatted output
print(formatted_text)
