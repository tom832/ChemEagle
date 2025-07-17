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

def merge_sentences(sentences):
    """
    合并一个句子片段列表为一个连贯的段落字符串。
    """
    # 去除每条片段前后空白，并剔除空串
    cleaned = [s.strip() for s in sentences if s.strip()]
    # 用空格拼接，恢复成完整段落
    paragraph = [" ".join(cleaned)]
    return paragraph


def extract_reactions_from_text_in_image(image_path: str) -> dict:
    """
    从化学反应图像中提取文本并识别反应。

    参数：
      image_path: 图像文件路径

    返回：
      {
        'raw_text': OCR 提取的完整文本（str),
        'paragraph': 合并后的段落文本 (str),
        'reactions': RxnExtractor 输出的反应列表 (list)
      }
    """
    # 模型目录和设备参数（可按需修改）
    model_dir = "./cre_models_v0.1"
    device = "cpu"

    # 1. OCR 提取文本
    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)

    # 2. 将多行文本合并为单段落
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    paragraph = " ".join(lines)

    # 3. 初始化化学反应提取器
    use_cuda = (device.lower() == "cuda")
    rxn_extractor = RxnExtractor(model_dir, use_cuda=use_cuda)

    # 4. 提取反应（注意 get_reactions 需要列表输入）
    reactions = rxn_extractor.get_reactions([paragraph])

    return reactions 

def NER_from_text_in_image(image_path: str) -> dict:
    # 模型目录和设备参数（可按需修改）
    model_dir = "./cre_models_v0.1"
    device = "cpu"

    # 1. OCR 提取文本
    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)

    # 2. 将多行文本合并为单段落
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    paragraph = " ".join(lines)

    # 3. 初始化化学反应提取器
    use_cuda = (device.lower() == "cuda")
    rxn_extractor = RxnExtractor(model_dir, use_cuda=use_cuda)

    # 4. 提取反应（注意 get_reactions 需要列表输入）
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
        api_version="2024-06-01",
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