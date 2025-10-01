import sys
import torch
import json
from chemietoolkit import ChemIEToolkit
import cv2
from PIL import Image
import json
import sys
import torch
from rxnim import RxnIM
import json
import sys
import torch
import json 
from molnextr.chemistry import _convert_graph_to_smiles
import base64
import torch
import json
from PIL import Image
import numpy as np
from chemietoolkit import ChemIEToolkit, utils
from openai import AzureOpenAI
import os





ckpt_path = "./pix2seq_reaction_full.ckpt"
model1 = RxnIM(ckpt_path, device=torch.device('cpu'))
device = torch.device('cpu')
model = ChemIEToolkit(device=torch.device('cpu'))

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Please set API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")

def get_multi_molecular(image_path: str) -> list:
    '''Returns a list of reactions extracted from the image.'''
    # 打开图像文件
    image = Image.open(image_path).convert('RGB')
    
    # 将图像作为输入传递给模型
    coref_results = model.extract_molecule_corefs_from_figures([image])
    #print(f"coref_results:{coref_results}")
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs']: #'atoms'
                bbox.pop(key, None)  # 安全地移除键
    #print(json.dumps(coref_results))
    # 返回反应列表，使用 json.dumps 进行格式化
    
    return json.dumps(coref_results)

def get_multi_molecular_text_to_correct(image_path: str) -> list:
    '''Returns a list of reactions extracted from the image.'''
    # 打开图像文件
    image = Image.open(image_path).convert('RGB')
    
    # 将图像作为输入传递给模型
    coref_results = model.extract_molecule_corefs_from_figures([image])
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["category", "bbox", "molfile", "symbols", 'atoms', "bonds", 'category_id', 'score', 'corefs']: #'atoms'
                bbox.pop(key, None)  # 安全地移除键
    #print(json.dumps(coref_results))
    # 返回反应列表，使用 json.dumps 进行格式化
    
    return json.dumps(coref_results)

def get_multi_molecular_text_to_correct_withatoms(image_path: str) -> list:
    '''Returns a list of reactions extracted from the image.'''
    # 打开图像文件
    image = Image.open(image_path).convert('RGB')
    
    # 将图像作为输入传递给模型
    coref_results = model.extract_molecule_corefs_from_figures([image])
    for item in coref_results:
        for bbox in item.get("bboxes", []):
            for key in ["coords","edges","molfile", 'atoms', "bonds", 'category_id', 'score', 'corefs']: #'atoms'
                bbox.pop(key, None)  # 安全地移除键
    #print(json.dumps(coref_results))
    # 返回反应列表，使用 json.dumps 进行格式化
    return json.dumps(coref_results)






def process_reaction_image_with_multiple_products_and_text(image_path: str) -> dict:
    """


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
            'name': 'get_multi_molecular_text_to_correct_withatoms',
            'description': 'Extracts the SMILES string, the symbols set, and the text coref of all molecular images in a table-reaction image and ready to be correct.',
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
    with open('./prompt/prompt_getmolecular.txt', 'r') as prompt_file:
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
        'get_multi_molecular_text_to_correct_withatoms': get_multi_molecular_text_to_correct_withatoms,
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
    gpt_output = [json.loads(response.choices[0].message.content)]


    def get_multi_molecular(image_path: str) -> list:
        '''Returns a list of reactions extracted from the image.'''
        # 打开图像文件
        image = Image.open(image_path).convert('RGB')
        
        # 将图像作为输入传递给模型
        coref_results = model.extract_molecule_corefs_from_figures([image])
        return coref_results

    
    coref_results = get_multi_molecular(image_path)


    def update_symbols_in_atoms(input1, input2):
        """
        用 input1 中更新后的 'symbols' 替换 input2 中对应 bboxes 的 'symbols'，并同步更新 'atoms' 的 'atom_symbol'。
        假设 input1 和 input2 的结构一致。
        """
        for item1, item2 in zip(input1, input2):
            bboxes1 = item1.get('bboxes', [])
            bboxes2 = item2.get('bboxes', [])
            
            if len(bboxes1) != len(bboxes2):
                print("Warning: Mismatched number of bboxes!")
                continue

            for bbox1, bbox2 in zip(bboxes1, bboxes2):
                # 更新 symbols
                if 'symbols' in bbox1:
                    bbox2['symbols'] = bbox1['symbols']  # 更新 symbols
                
                # 更新 atoms 的 atom_symbol
                if 'symbols' in bbox1 and 'atoms' in bbox2:
                    symbols = bbox1['symbols']
                    atoms = bbox2.get('atoms', [])
                    
                    # 确保 symbols 和 atoms 的长度一致
                    if len(symbols) != len(atoms):
                        print(f"Warning: Mismatched symbols and atoms in bbox {bbox1.get('bbox')}!")
                        continue

                    for atom, symbol in zip(atoms, symbols):
                        atom['atom_symbol'] = symbol  # 更新 atom_symbol

        return input2


    input2_updated = update_symbols_in_atoms(gpt_output, coref_results)





    def update_smiles_and_molfile(input_data, conversion_function):
        """
        使用更新后的 'symbols'、'coords' 和 'edges' 调用 `conversion_function` 生成新的 'smiles' 和 'molfile'，
        并替换到原数据结构中。
        
        参数:
        - input_data: 包含 bboxes 的嵌套数据结构
        - conversion_function: 函数，接受 'coords', 'symbols', 'edges' 并返回 (new_smiles, new_molfile, _)
        
        返回:
        - 更新后的数据结构
        """
        for item in input_data:
            for bbox in item.get('bboxes', []):
                # 检查必需的键是否存在
                if all(key in bbox for key in ['coords', 'symbols', 'edges']):
                    coords = bbox['coords']
                    symbols = bbox['symbols']
                    edges = bbox['edges']
                    
                    # 调用转换函数生成新的 'smiles' 和 'molfile'
                    new_smiles, new_molfile, _ = conversion_function(coords, symbols, edges)
                    #print(f"    Generated 'smiles': {new_smiles}")
            
                    # 替换旧的 'smiles' 和 'molfile'
                    bbox['smiles'] = new_smiles
                    bbox['molfile'] = new_molfile

        return input_data

    updated_data = update_smiles_and_molfile(input2_updated, _convert_graph_to_smiles)

    return updated_data

    
    






def process_reaction_image_with_multiple_products_and_text_correctR(image_path: str) -> dict:
    """


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
            'name': 'get_multi_molecular_text_to_correct_withatoms',
            'description': 'Extracts the SMILES string, the symbols set, and the text coref of all molecular images in a table-reaction image and ready to be correct.',
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
    with open('./prompt/prompt_getmolecular_correctR.txt', 'r') as prompt_file:
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
        'get_multi_molecular_text_to_correct_withatoms': get_multi_molecular_text_to_correct_withatoms,
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
    gpt_output = [json.loads(response.choices[0].message.content)]
    print(f"gpt_output_mol:{gpt_output}")


    def get_multi_molecular(image_path: str) -> list:
        '''Returns a list of reactions extracted from the image.'''
        # 打开图像文件
        image = Image.open(image_path).convert('RGB')
        
        # 将图像作为输入传递给模型
        coref_results = model.extract_molecule_corefs_from_figures([image])
        return coref_results

    
    coref_results = get_multi_molecular(image_path)


    def update_symbols_in_atoms(input1, input2):
        """
        用 input1 中更新后的 'symbols' 替换 input2 中对应 bboxes 的 'symbols'，并同步更新 'atoms' 的 'atom_symbol'。
        假设 input1 和 input2 的结构一致。
        """
        for item1, item2 in zip(input1, input2):
            bboxes1 = item1.get('bboxes', [])
            bboxes2 = item2.get('bboxes', [])
            
            if len(bboxes1) != len(bboxes2):
                print("Warning: Mismatched number of bboxes!")
                continue

            for bbox1, bbox2 in zip(bboxes1, bboxes2):
                # 更新 symbols
                if 'symbols' in bbox1:
                    bbox2['symbols'] = bbox1['symbols']  # 更新 symbols
                
                # 更新 atoms 的 atom_symbol
                if 'symbols' in bbox1 and 'atoms' in bbox2:
                    symbols = bbox1['symbols']
                    atoms = bbox2.get('atoms', [])
                    
                    # 确保 symbols 和 atoms 的长度一致
                    if len(symbols) != len(atoms):
                        print(f"Warning: Mismatched symbols and atoms in bbox {bbox1.get('bbox')}!")
                        continue

                    for atom, symbol in zip(atoms, symbols):
                        atom['atom_symbol'] = symbol  # 更新 atom_symbol

        return input2


    input2_updated = update_symbols_in_atoms(gpt_output, coref_results)





    def update_smiles_and_molfile(input_data, conversion_function):
        """
        使用更新后的 'symbols'、'coords' 和 'edges' 调用 `conversion_function` 生成新的 'smiles' 和 'molfile'，
        并替换到原数据结构中。
        
        参数:
        - input_data: 包含 bboxes 的嵌套数据结构
        - conversion_function: 函数，接受 'coords', 'symbols', 'edges' 并返回 (new_smiles, new_molfile, _)
        
        返回:
        - 更新后的数据结构
        """
        for item in input_data:
            for bbox in item.get('bboxes', []):
                # 检查必需的键是否存在
                if all(key in bbox for key in ['coords', 'symbols', 'edges']):
                    coords = bbox['coords']
                    symbols = bbox['symbols']
                    edges = bbox['edges']
                    
                    # 调用转换函数生成新的 'smiles' 和 'molfile'
                    new_smiles, new_molfile, _ = conversion_function(coords, symbols, edges)
                    #print(f"    Generated 'smiles': {new_smiles}")
            
                    # 替换旧的 'smiles' 和 'molfile'
                    bbox['smiles'] = new_smiles
                    bbox['molfile'] = new_molfile

        return input_data

    updated_data = update_smiles_and_molfile(input2_updated, _convert_graph_to_smiles)
    print(f"mol_agent_output:{updated_data}")

    return updated_data




def process_reaction_image_with_multiple_products_and_text_correctmultiR(image_path: str) -> dict:
    """


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
            'name': 'get_multi_molecular_text_to_correct_withatoms',
            'description': 'Extracts the SMILES string, the symbols set, and the text coref of all molecular images in a table-reaction image and ready to be correct.',
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
    with open('./prompt/prompt_getmolecular_correctmultiR.txt', 'r') as prompt_file:
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
        'get_multi_molecular_text_to_correct_withatoms': get_multi_molecular_text_to_correct_withatoms,
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
    gpt_output = [json.loads(response.choices[0].message.content)]
    print(f"gpt_output_mol:{gpt_output}")


    def get_multi_molecular(image_path: str) -> list:
        '''Returns a list of reactions extracted from the image.'''
        # 打开图像文件
        image = Image.open(image_path).convert('RGB')
        
        # 将图像作为输入传递给模型
        coref_results = model.extract_molecule_corefs_from_figures([image])
        return coref_results

    
    coref_results = get_multi_molecular(image_path)


    def update_symbols_and_corefs(gpt_outputs, coref_results):
        results = []
        for item1, item2 in zip(gpt_outputs, coref_results):
            orig_bboxes = item2.get('bboxes', [])
            orig_corefs = item2.get('corefs', [])
            # 1. 构造新的bboxes（严格用同bbox作为模板）
            coord2idx = {tuple(bb['bbox']): i for i, bb in enumerate(orig_bboxes)}
            new_bboxes = []
            for bb1 in item1.get('bboxes', []):
                coord = tuple(bb1['bbox'])
                if coord in coord2idx:
                    bb_template = orig_bboxes[coord2idx[coord]]
                else:
                    raise ValueError(f"扩展mol时未找到bbox {coord} 的原始模板！")
                bb_new = copy.deepcopy(bb_template)
                if 'symbols' in bb1:
                    bb_new['symbols'] = bb1['symbols']
                    if 'atoms' in bb_new:
                        for atom, sym in zip(bb_new['atoms'], bb1['symbols']):
                            atom['atom_symbol'] = sym
                if 'text' in bb1:
                    bb_new['text'] = bb1['text']
                bb_new['bbox'] = bb1['bbox']
                new_bboxes.append(bb_new)
            
            # 2. 构建corefs（所有同类mol都和扩展后对应的label索引分组）
            # 步骤：找出原组里mol的所有新索引，以及label的新索引，按原corefs分组生成新组
            coord2new_idxs = {}
            for idx, bb in enumerate(new_bboxes):
                coord = tuple(bb['bbox'])
                coord2new_idxs.setdefault(coord, []).append(idx)
            new_corefs = []
            for group in orig_corefs:
                # 假设group = [mol_idx, idt_idx] 或 [mol_idx1, mol_idx2, ..., idt_idx]
                label_idx = group[-1]
                label_coord = tuple(orig_bboxes[label_idx]['bbox'])
                new_label_idx = coord2new_idxs[label_coord][-1]  # label只会有一个
                # 所有mol的扩展后新索引
                for mol_idx in group[:-1]:
                    mol_coord = tuple(orig_bboxes[mol_idx]['bbox'])
                    for new_mol_idx in coord2new_idxs[mol_coord]:
                        new_corefs.append([new_mol_idx, new_label_idx])
            # 3. 装配结构
            new_item = copy.deepcopy(item2)
            new_item['bboxes'] = new_bboxes
            new_item['corefs'] = new_corefs
            results.append(new_item)
        return results


    input2_updated = update_symbols_and_corefs(gpt_output, coref_results)





    def update_smiles_and_molfile(input_data, conversion_function):
        """
        使用更新后的 'symbols'、'coords' 和 'edges' 调用 `conversion_function` 生成新的 'smiles' 和 'molfile'，
        并替换到原数据结构中。
        
        参数:
        - input_data: 包含 bboxes 的嵌套数据结构
        - conversion_function: 函数，接受 'coords', 'symbols', 'edges' 并返回 (new_smiles, new_molfile, _)
        
        返回:
        - 更新后的数据结构
        """
        for item in input_data:
            for bbox in item.get('bboxes', []):
                # 检查必需的键是否存在
                if all(key in bbox for key in ['coords', 'symbols', 'edges']):
                    coords = bbox['coords']
                    symbols = bbox['symbols']
                    edges = bbox['edges']
                    
                    # 调用转换函数生成新的 'smiles' 和 'molfile'
                    new_smiles, new_molfile, _ = conversion_function(coords, symbols, edges)
                    #print(f"    Generated 'smiles': {new_smiles}")
            
                    # 替换旧的 'smiles' 和 'molfile'
                    bbox['smiles'] = new_smiles
                    bbox['molfile'] = new_molfile

        return input_data

    updated_data = update_smiles_and_molfile(input2_updated, _convert_graph_to_smiles)
    print(f"mol_agent_output:{updated_data}")

    return updated_data
