from PIL import Image
import pytesseract
from chemrxnextractor import RxnExtractor
from openai import AzureOpenAI
model_dir = "./cre_models_v0.1"
rxn_extractor = RxnExtractor(model_dir)
import json
import torch
from chemiener import ChemNER
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download("Ozymandias314/ChemNERCkpt", "best.ckpt")
model2 = ChemNER(ckpt_path, device=torch.device('cpu'))
import base64
import os

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Please set API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")

def merge_sentences(sentences):

    cleaned = [s.strip() for s in sentences if s.strip()]
    paragraph = [" ".join(cleaned)]
    return paragraph


def extract_reactions_from_text_in_image(image_path: str) -> dict:


    model_dir = "./cre_models_v0.1"
    device = "cpu"

    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    paragraph = " ".join(lines)

    use_cuda = (device.lower() == "cuda")
    rxn_extractor = RxnExtractor(model_dir, use_cuda=use_cuda)
）
    reactions = rxn_extractor.get_reactions([paragraph])

    return reactions 

def NER_from_text_in_image(image_path: str) -> dict:
    model_dir = "./cre_models_v0.1"
    device = "cpu"

    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    paragraph = " ".join(lines)
    use_cuda = (device.lower() == "cuda")
    rxn_extractor = RxnExtractor(model_dir, use_cuda=use_cuda)
）
    predictions = model2.predict_strings([paragraph])

    return predictions 




def text_extraction_agent(image_path: str) -> dict:
    """
    Agent that calls two tools:
      1) extract_reactions_from_text_in_image
      2) NER_from_text_in_image
    to perform OCR, reaction extraction, and chemical NER on a single image.
    Returns a merged JSON result.
    """
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )

    # Encode image as Base64
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    # Define tools for the agent
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_reactions_from_text_in_image",
                "description": "OCR image and extract chemical reactions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "NER_from_text_in_image",
                "description": "OCR image and perform chemical named entity recognition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"]
                }
            }
        }
    ]

    # Prompt instructing to call both tools
    prompt = (
        "Please call the following two tools in order: "
        "extract_reactions_from_text_in_image and NER_from_text_in_image. "
        "First, extract all text from the image and merge into a single paragraph. "
        "Second, identify chemical reactions, extracting reactants, products, reagents, and conditions. "
        "Third, perform chemical named entity recognition to label molecules, reagents, and other chemical entities. "
        "Return a JSON object with keys: reactions, NER_results."
    )

    messages = [
        {"role": "system", "content": "You are an expert assistant for chemical text analysis."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        }
    ]

    # First API call: let GPT decide which tools to invoke
    response1 = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        temperature=0,
        response_format={"type": "json_object"}
    )

    # Execute each requested tool
    tool_calls = response1.choices[0].message.tool_calls
    tool_results_msgs = []
    for call in tool_calls:
        name = call.function.name
        if name == "extract_reactions_from_text_in_image":
            result = extract_reactions_from_text_in_image(image_path)
        elif name == "NER_from_text_in_image":
            result = NER_from_text_in_image(image_path)
        else:
            continue
        tool_results_msgs.append({
            "role": "tool",
            "tool_name": name,
            "content": json.dumps(result)
        })

    # Second API call: pass tool outputs back to GPT for final response
    response2 = client.chat.completions.create(
        model="gpt-4o",
        messages=messages + tool_results_msgs,
        temperature=0,
        response_format={"type": "json_object"}
    )

    return json.loads(response2.choices[0].message.content)
