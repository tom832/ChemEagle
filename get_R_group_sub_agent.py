import sys
import torch
import json
from chemietoolkit import ChemIEToolkit,utils
import cv2
from openai import AzureOpenAI
import numpy as np
from PIL import Image
import json
from get_molecular_agent import process_reaction_image_with_multiple_products_and_text_correctR, process_reaction_image_with_multiple_products_and_text_correctmultiR
from get_reaction_agent import get_reaction_withatoms_correctR
import sys
from rxnim import RxnScribe
import json
import base64
model = ChemIEToolkit(device=torch.device('cpu')) 
ckpt_path = "./pix2seq_reaction_full.ckpt"
model1 = RxnScribe(ckpt_path, device=torch.device('cpu'))
device = torch.device('cpu')
import base64
import torch
import json
from PIL import Image
import numpy as np
from openai import AzureOpenAI
import copy
from molnextr.chemistry import _convert_graph_to_smiles 
import os


API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Please set API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")




def parse_coref_data_with_fallback(data):
    bboxes = data["bboxes"]
    corefs = data["corefs"]
    paired_indices = set()

    # 先处理有 coref 配对的
    results = []
    for idx1, idx2 in corefs:
        smiles_entry = bboxes[idx1] if "smiles" in bboxes[idx1] else bboxes[idx2]
        text_entry = bboxes[idx2] if "text" in bboxes[idx2] else bboxes[idx1]

        smiles = smiles_entry.get("smiles", "")
        bbox= smiles_entry.get("bbox", ())
        texts = text_entry.get("text", [])

        results.append({
            "smiles": smiles,
            "texts": texts,
            "bbox": bbox
        })

        # 记录下哪些 SMILES 被配对过了
        paired_indices.add(idx1)
        paired_indices.add(idx2)

    # 处理未配对的 SMILES（补充进来）
    for idx, entry in enumerate(bboxes):
        if "smiles" in entry and idx not in paired_indices:
            results.append({
                "smiles": entry["smiles"],
                "texts": ["There is no label or failed to detect, please recheck the image again"],
                "bbox": entry["bbox"],
            })

    return results

def parse_coref_data_with_fallback_with_box(data):
    bboxes = data["bboxes"]
    corefs = data["corefs"]
    paired_indices = set()

    # 先处理有 coref 配对的
    results = []
    for idx1, idx2 in corefs:
        smiles_entry = bboxes[idx1] if "smiles" in bboxes[idx1] else bboxes[idx2]
        text_entry = bboxes[idx2] if "text" in bboxes[idx2] else bboxes[idx1]

        smiles = smiles_entry.get("smiles", "")
        bboxes = smiles_entry.get("bbox", [])
        texts = text_entry.get("text", [])

        results.append({
            "smiles": smiles,
            "texts": texts,
            "bbox": bboxes
        })

        # 记录下哪些 SMILES 被配对过了
        paired_indices.add(idx1)
        paired_indices.add(idx2)

    # 处理未配对的 SMILES（补充进来）
    for idx, entry in enumerate(bboxes):
        if "smiles" in entry and idx not in paired_indices:
            results.append({
                "smiles": entry["smiles"],
                "texts": ["There is no label or failed to detect, please recheck the image again"],
                "bbox": entry["bbox"],
            })

    return results





############################### MOl
_process_multi_molecular_cache = {}

def get_cached_multi_molecular(image_path: str):
    """
    只会对同一个 image_path 真正调用一次
    process_reaction_image_with_multiple_products_and_text_correctR
    并缓存结果。
    """
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    if image_path not in _process_multi_molecular_cache:
        ##print(f"[get_cached_multi_molecular] Processing image: {image_path}")
        _process_multi_molecular_cache[image_path] = (
            process_reaction_image_with_multiple_products_and_text_correctmultiR(image_path)
            ################################model.extract_molecule_corefs_from_figures([image])#############################################################################################
            )
        ##print(f"original output: {model.extract_molecule_corefs_from_figures([image])}")
    return _process_multi_molecular_cache[image_path]



# def get_multi_molecular_text_to_correct(image_path: str) -> list:
#     '''Returns a list of reactions extracted from the image.'''
#     # 打开图像文件
#     image = Image.open(image_path).convert('RGB')
    
#     # 将图像作为输入传递给模型
#     #coref_results = process_reaction_image_with_multiple_products_and_text_correctR(image_path)
#     coref_results = model.extract_molecule_corefs_from_figures([image])
#     for item in coref_results:
#         for bbox in item.get("bboxes", []):
#             for key in ["category", "bbox", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
#                 bbox.pop(key, None)  # 安全地移除键

#     data = coref_results[0]
#     parsed = parse_coref_data_with_fallback(data)
    # #print(f"coref_results:{json.dumps(parsed)}")
    # #return json.dumps(parsed)
    # return parsed

def get_multi_molecular_text_to_correct(image_path: str) -> list:
    """
    GPT-4o 注册的 tool。内部不再直接调用二级 Agent，
    而是复用缓存过的结果。
    """
    coref_results = copy.deepcopy(get_cached_multi_molecular(image_path))

    # 按需删掉不想返回给 LLM 的字段
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in [
                "category", "molfile", "symbols",
                "atoms", "bonds", "category_id", "score", "corefs",
                "coords", "edges"
            ]:
                bbox.pop(key, None)

    # 假设 parse_coref_data_with_fallback 需要传入单个 dict
    parsed = parse_coref_data_with_fallback(coref_results[0])
    ##print(f"[get_multi_molecular_text_to_correct] parsed: {json.dumps(parsed)}")
    return parsed

    



def get_multi_molecular_full(image_path: str) -> list:
    '''Returns a list of reactions extracted from the image.'''
    # 打开图像文件
    image = Image.open(image_path).convert('RGB')
    
    # 将图像作为输入传递给模型
    #coref_results = process_reaction_image_with_multiple_products_and_text_correctR(image_path)
    coref_results = model.extract_molecule_corefs_from_figures([image])
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
                bbox.pop(key, None)  # 安全地移除键

    data = coref_results[0]
    parsed = parse_coref_data_with_fallback(data)

    
    ##print(f"coref_results:{json.dumps(parsed)}")
    #return json.dumps(parsed)
    return parsed
#get_multi_molecular_text_to_correct('./acs.joc.2c00176 example 1.png')

_raw_results_cache = {}

def get_cached_raw_results(image_path: str):
    """
    调用一次 get_reaction_withatoms_correctR 并缓存结果，
    后续复用同一份 raw_results。
    """
    if image_path not in _raw_results_cache:
        #print(f"[get_cached_raw_results] Processing image: {image_path}")
        _raw_results_cache[image_path] = get_reaction_withatoms_correctR(image_path)
        ###############################_raw_results_cache[image_path]= model1.predict_image_file(image_path, molscribe=True, ocr=True)####################################################################
    return _raw_results_cache[image_path]


# ----------------------------------------
# 工具函数：基于 raw_pred 构造精简输出
# ----------------------------------------
def get_reaction_from_raw(raw_pred: dict) -> dict:
    """
    Returns a structured dictionary of reactions extracted from the raw prediction,
    """
    structured = {}
    for section in ['reactants', 'conditions', 'products']:
        if section in raw_pred:
            structured[section] = []
            for item in raw_pred[section]:
                if section in ('reactants', 'products'):
                    structured[section].append({
                        "smiles": item.get("smiles", ""),
                        "bbox":   item.get("bbox",   [])
                    })
                else:  # conditions
                    structured[section].append({
                        "text":   item.get("text",   []),
                        "bbox":   item.get("bbox",   []),
                        "smiles": item.get("smiles", [])
                    })
    return structured

# ----------------------------------------
# LLM 工具：get_reaction
# ----------------------------------------
def get_reaction(image_path: str) -> dict:
    """    
    Returns a structured dictionary of reactions extracted from the image,
    """
    # 复用缓存的 raw_results
    raw_pred = get_cached_raw_results(image_path)[0]
    return get_reaction_from_raw(raw_pred)




def get_reaction_full(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image, 
    including only reactants, conditions, and products with their smiles, bbox, or text.
    '''
    image_file = image_path
    raw_prediction = model1.predict_image_file(image_file, molscribe=True, ocr=True)
    #raw_prediction = get_reaction_withatoms_correctR(image_path)
    return raw_prediction

def get_full_reaction(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image = Image.open(image_path).convert('RGB')
    image_file = image_path
    raw_prediction = model1.predict_image_file(image_file, molscribe=True, ocr=True)
    ####################raw_prediction = get_reaction_withatoms_correctR(image_path)###############################################################################################
    for reaction in raw_prediction:
        for section in ("reactants", "products", "conditions"):
            for entry in reaction.get(section, []):
                # 1) 保留 coords 三位小数
                coords = entry.get("coords")
                if isinstance(coords, list):
                    entry["coords"] = [
                        [round(val, 3) for val in point]
                        for point in coords
                    ]
                # 2) 删除不需要的字段
                for key in ("molfile", "atoms", "bonds"):
                    entry.pop(key, None)

    #raw_prediction =json.dumps(raw_prediction)
    print(f"raw_prediction:{raw_prediction}")
    coref_results = model.extract_molecule_corefs_from_figures([image])
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
                bbox.pop(key, None)  # 安全地移除键

    data = coref_results[0]
    parsed = parse_coref_data_with_fallback(data)

    combined_result = {
        "reaction_prediction": raw_prediction,  # 是个list
        "molecule_coref": parsed               # 结构化分子识别结果
    }
    print(f"combined_result:{combined_result}")
    return combined_result



def process_reaction_image_with_product_variant_R_group(image_path: str) -> dict:
    """
    输入化学反应图像路径，通过 GPT 模型和 OpenChemIE 提取反应信息并返回整理后的反应数据。

    Args:
        image_path (str): 图像文件路径。

    Returns:
        dict: 整理后的反应数据，包括反应物、产物和反应模板。
    """
 
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )

    # 加载图像并编码为 Base64
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    # GPT 工具调用配置
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'get_multi_molecular_text_to_correct',
                'description': 'Extracts the SMILES string and text coref from molecular images.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'image_path': {
                            'type': 'string',
                            'description': 'Path to the reaction image.'
                        }
                    },
                    'required': ['image_path'],
                    'additionalProperties': False
                }
            }
        },
        {
        'type': 'function',
        'function': {
            'name': 'get_reaction',
            'description': 'Get a list of reactions from a reaction image. A reaction contains data of the reactants, conditions, and products.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
            },
    ]

    # 提供给 GPT 的消息内容
    with open('./prompt/prompt.txt', 'r') as prompt_file:
        prompt = prompt_file.read()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
            ]
        }
    ]

    # 调用 GPT 接口
    response = client.chat.completions.create(
    model = 'gpt-4o',
    temperature = 0,
    response_format={ 'type': 'json_object' },
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': prompt
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/png;base64,{base64_image}'
                    }
                }
            ]},
    ],
    tools = tools)
    
# Step 1: 工具映射表
    TOOL_MAP = {
        'get_multi_molecular_text_to_correct': get_multi_molecular_text_to_correct,
        'get_reaction': get_reaction
    }

    # Step 2: 处理多个工具调用
    tool_calls = response.choices[0].message.tool_calls
    results = []

    # 遍历每个工具调用
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = tool_call.function.arguments
        tool_call_id = tool_call.id
        
        tool_args = json.loads(tool_arguments)
        
        if tool_name in TOOL_MAP:
            # 调用工具并获取结果
            tool_result = TOOL_MAP[tool_name](image_path)
        else:
            raise ValueError(f"Unknown tool called: {tool_name}")
        
        # 保存每个工具调用结果
        results.append({
            'role': 'tool',
            'content': json.dumps({
                'image_path': image_path,
                f'{tool_name}':(tool_result),
            }),
            'tool_call_id': tool_call_id,
        })
    #print(f"tool_results:{tool_result}")


# Prepare the chat completion payload
    completion_payload = {
        'model': 'gpt-4o',
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': prompt
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{base64_image}'
                        }
                    }
                ]
            },
            response.choices[0].message,
            *results
            ],
    }

# Generate new response
    response = client.chat.completions.create(
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        response_format={ 'type': 'json_object' },
        temperature=0
    )


    
    # 获取 GPT 生成的结果
    gpt_output = json.loads(response.choices[0].message.content)
    print("R_group_agent_output:", gpt_output)
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

 
    #coref_results = model.extract_molecule_corefs_from_figures([image_np])
    #coref_results = process_reaction_image_with_multiple_products_and_text_correctR(image_path)
    coref_results =get_cached_multi_molecular(image_path)


    # reaction_results = model.extract_reactions_from_figures([image_np])
    #reaction_results = get_reaction_withatoms_correctR(image_path)[0]
    raw_results  = get_cached_raw_results(image_path)
    reaction_results = raw_results[0]
    
    reaction = {
    "reactants": reaction_results.get('reactants', []),
    "conditions": reaction_results.get('conditions', []),
    "products": reaction_results.get('products', [])
    }
    reaction_results = [{"reactions": [reaction]}]
    #print(reaction_results)
    

    # 定义更新工具输出的函数
    def extract_smiles_details(smiles_data, raw_details):
        smiles_details = {}
        for smiles in smiles_data:
            for detail in raw_details:
                for bbox in detail.get('bboxes', []):
                    if bbox.get('smiles') == smiles:
                        smiles_details[smiles] = {
                            'category': bbox.get('category'),
                            'bbox': bbox.get('bbox'),
                            'category_id': bbox.get('category_id'),
                            'score': bbox.get('score'),
                            'molfile': bbox.get('molfile'),
                            'atoms': bbox.get('atoms'),
                            'bonds': bbox.get('bonds'),
                        }
                        break
        return smiles_details

# 获取结果
    smiles_details = extract_smiles_details(gpt_output, coref_results)
    #print('smiles_details:', smiles_details)

    reactants_array = []
    products = []

    for reactant in reaction_results[0]['reactions'][0]['reactants']:
        if 'smiles' in reactant:
            #print(f"SMILES:{reactant['smiles']}")
            ##print(reactant)
            reactants_array.append(reactant['smiles'])

    for product in reaction_results[0]['reactions'][0]['products']:
        ##print(product['smiles'])
        ##print(product)
        products.append(product['smiles'])
    # 输出结果
    #import p#print
    #p#print.p#print(smiles_details)

        # 整理反应数据
    backed_out = utils.backout_without_coref(reaction_results, coref_results, gpt_output, smiles_details, model.molscribe)
    backed_out.sort(key=lambda x: x[2])
    extracted_rxns = {}
    for reactants, products_, label in backed_out:
        extracted_rxns[label] = {'reactants': reactants, 'products': products_}
    
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs',"coords","edges"]: #'atoms'
                bbox.pop(key, None)  # 安全地移除键

    data = coref_results[0]
    parsed = parse_coref_data_with_fallback(data)
    
    toadd = {
        "reaction_template": {
            "reactants": reactants_array,
            "products": products
        },
        "reactions": extracted_rxns,
        "original_molecule_list": parsed
    }

# 按标签排序
    sorted_keys = sorted(toadd["reactions"].keys())
    toadd["reactions"] = {i: toadd["reactions"][i] for i in sorted_keys}
    print(f"str_R_group_agent_output:{toadd}")
    return toadd



def process_reaction_image_with_table_R_group(image_path: str) -> dict:

    client = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )

    # 加载图像并编码为 Base64
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)
    with open('./prompt/prompt_reaction_withR.txt', 'r') as prompt_file:
        prompt = prompt_file.read()
    tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_full_reaction',
            'description': 'Get a list of reactions from a reaction image. A reaction contains data of the reactants, conditions, and products.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
    },
    ]

    
    response = client.chat.completions.create(
    model = 'gpt-4o',
    temperature = 0,
    response_format={ 'type': 'json_object' },
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': prompt
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/png;base64,{base64_image}'
                    }
                }
            ]},
    ],
    tools = tools,
    )

    
    tool_call = response.choices[0].message.tool_calls[0]
    tool_name = tool_call.function.name  # 修改此处
    tool_arguments = tool_call.function.arguments  # 新增此处
    tool_call_id = tool_call.id

    tool_args = json.loads(tool_arguments)
    #image_path = tool_args.get('image_path', image_path)  # 使用模型提供的 image_path

    if tool_name == 'get_full_reaction':
        tool_result = get_full_reaction(image_path)

    else:
        raise ValueError(f"Unknown tool called: {tool_name}")
    #print(tool_result)

    # 构建工具调用结果消息
    function_call_result_message = {
        'role': 'tool',
        'content': json.dumps({
            'image_path': image_path,
            f'{tool_name}':(tool_result),
    }),
        'tool_call_id': tool_call_id,
    }


    completion_payload = {
        'model': 'gpt-4o',
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': prompt
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{base64_image}'
                        }
                    }
                ]
            },
            response.choices[0].message,
            function_call_result_message,
        ],
    }

    # Generate new response
    response = client.chat.completions.create(
        model=completion_payload["model"],
        messages=completion_payload["messages"],
        response_format={ 'type': 'json_object' },
        temperature=0
    )

    #print(response)   


    def replace_symbols_and_generate_smiles(input1, input2):
        """
        通用函数，用于将输入2中的symbols替换到输入1中，并生成新的SMILES。
        返回的结果保持特定格式，不包含初始的反应数据。
        
        参数:
        input1: 包含reactants和products的初始输入数据
        input2: 包含不同反应的symbols信息的数据

        返回:
        一个新的包含每个reaction的字典，包含reaction_id、reactants和products。
        """
        
        reactions_output = {"reactions": []}  # 存储最终的反应输出
        
        # 遍历 input2 中的每个 reaction
        for reaction in input2['reactions']:
            reaction_id = reaction['reaction_id']
            
            # 构建新的 reaction 字典
            new_reaction = {"reaction_id": reaction_id, "reactants": [], "conditions":[], "products": [], "additional_info": []}

            for j, reactant in enumerate(reaction['reactants']):
                original_reactant = input1['reactants'][j]  # 对应的 reactant 模板
                ##print(original_reactant)
                new_symbols_reactant = reactant['symbols']  # 替换为reaction中的symbols
                new_smiles_reactant, __, __ = _convert_graph_to_smiles(original_reactant['coords'], new_symbols_reactant, original_reactant['edges'])  # 生成新的SMILES
                
                new_reactant = {
                    #"category": original_reactant['category'],
                    #"bbox": original_reactant['bbox'],
                    #"category_id": original_reactant['category_id'],
                    "smiles": new_smiles_reactant,
                    #"coords": original_reactant['coords'],
                    "symbols": new_symbols_reactant,
                    #"edges": original_reactant['edges']
                }
                new_reaction["reactants"].append(new_reactant)

            if 'conditions' in reaction:
                new_reaction['conditions'] = reaction['conditions']

            
            # 处理 products 中的每个分子
            for k, product in enumerate(reaction['products']):
                original_product = input1['products'][k]  # 对应的 product 模板
                new_symbols_product = product['symbols']  # 替换为reaction中的symbols
                new_smiles_product, __, __ = _convert_graph_to_smiles(original_product['coords'], new_symbols_product, original_product['edges'])  # 生成新的SMILES
                
                new_product = {
                    #"category": original_product['category'],
                    #"bbox": original_product['bbox'],
                    #"category_id": original_product['category_id'],
                    "smiles": new_smiles_product,
                    #"coords": original_product['coords'],
                    "symbols": new_symbols_product,
                    #"edges": original_product['edges']
                }
                new_reaction["products"].append(new_product)
            
            if 'additional_info' in reaction:
                new_reaction['additional_info'] = reaction['additional_info']

            reactions_output['reactions'].append(new_reaction)  

        return reactions_output
    

    reaction_preds = tool_result['reaction_prediction']
    if isinstance(reaction_preds, str):
        # 如果是字符串，就 parse
        tool_result_json = json.loads(reaction_preds)
    elif isinstance(reaction_preds, (dict, list)):
        # 已经是 dict 或 list，直接使用
        tool_result_json = reaction_preds
    else:
        raise TypeError(f"Unexpected tool_result type: {type(reaction_preds)}")

    input1 = tool_result_json[0]
    input2 = json.loads(response.choices[0].message.content) 
    updated_input = replace_symbols_and_generate_smiles(input1, input2)
    print(f"txt_R_group_agent_output:{updated_input}")
    return updated_input

