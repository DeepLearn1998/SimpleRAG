import fitz
import os
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv
import requests


# 加载 .env
load_dotenv()

def extract_text_from_pdf(pdf_path):
    """
    从 PDF 中逐页提取信息，并拼接为一个 text

    Args:
    pdf_path (str): PDF 文件路径

    Returns:
    str: 提取信息
    """
    # 打开 PDF 文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化一个空字符串，用于存储提取信息

    # 按照页码迭代 PDF
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # 获取页码
        text = page.get_text("text")  # 提取信息
        all_text += text  # all_text: 存储提取信息

    return all_text  # 输出提取到的信息


def chunk_text(text, n, overlap):
    """
    带有重叠部分的文档切片

    Args:
    text (str): 待切片文本
    n (int): 每片 chunk 中的字数
    overlap (int): chunks 之间的重叠字数

    Returns:
    List[str]: chunk list
    """
    chunks = []  # 初始化一个空的 list，来存储 chunks

    # 循环 text，步长为 n - overlap
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])  # 将每个 chunk 添加到 chunks 中

    return chunks

def create_embeddings(text, model="doubao-embedding-large-text-240915"):
    """
    使用指定的嵌入模型，对给定文本，生成对应的 embedding 结果

    Args:
    text (str): 待嵌入的文本
    model (str): 用于生成 embedding 的嵌入模型，默认采用 "doubao-embedding-large-text-240915"

    Returns:
    dict: embedding 结果
    """
    # 基于 base URL 和 API key 初始化 LLM
    client = OpenAI(
        api_key=os.environ.get("ARK_API_KEY"),  # 需在 .env 中替换为自己的 api key！！！
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    response = client.embeddings.create(
        model=model,
        input=text,
        encoding_format="float"
    )

    return response

def cosine_similarity(vec1, vec2):
    """
    计算两向量之间的余弦相似度

    Args:
    vec1 (np.ndarray): vector 1
    vec2 (np.ndarray): vector 2

    Returns:
    float: 两向量之间的余弦相似度
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, text_chunks, embeddings, k=5):
    """
    根据给定的查询和 embedding，在 chunks 中执行相似度检索

    Args:
    query (str): 用户查询
    text_chunks (List[str]): 待查询的 chunks
    embeddings (List[dict]): chunks 的 embeddings
    k (int): top k (默认为5)

    Returns:
    List[str]: top k 个与 query 最相关的 chunks
    """
    # 计算 query 的 embedding 结果
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []  # 初始化一个空的 list，存储相似度得分

    # 计算 query 和 chunks 之间的相似度得分
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))  # 根据 index 添加 similarity_score

    # 按照降序存储相似度得分
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    # 获取 top k 的 index
    top_indices = [index for index, _ in similarity_scores[:k]]

    return [text_chunks[index] for index in top_indices]

def generate_response(system_prompt, user_message, model="deepseek-v3-241226"):
    """
    LLM 根据 system prompt 和 user message 生成响应

    Args:
    system_prompt (str): 引导 AI 的系统提示词
    user_message (str): 来自用户查询的消息
    model (str): 用于生成回答的模型，默认采用 "deepseek-v3-241226"

    Returns:
    dict: LLM 生成的回答
    """
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    }
    headers ={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('ARK_API_KEY')}",
    }

    response = requests.request("POST", url, json=data, headers=headers)
    content = ""
    result = json.loads(response.text).get('choices', [])
    for r in result:
        content = r.get('message', {}).get('content', '')

    return content


if __name__ == "__main__":
    # PDF 文件路径
    pdf_path = "data/test_document.pdf"
    # 从 PDF 文件中提取信息
    extracted_text = extract_text_from_pdf(pdf_path)
    # 切片长度 200，重叠长度 50
    text_chunks = chunk_text(extracted_text, 200, 50)
    print("Number of text chunks:", len(text_chunks))  # 文档切片数量
    print("First text chunk:\n", text_chunks[0])  # 查看第一个 chunk

    # 生成 embedding
    response = create_embeddings(text_chunks)

    # 按照 json 格式加载验证集，理想情况下的问答对（list）：[{"question": "假设问题", "answer": "预期回答"}]
    with open('data/val.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取验证集中的第一个问题
    query = data[0]['question']
    print("Query:", query)

    # 相似度检索
    top_chunks = semantic_search(query, text_chunks, response.data, k=2)
    for i, chunk in enumerate(top_chunks):
        print(f"Context {i + 1}:\n{chunk}\n=====================================")

    # system prompt
    system_prompt = "你是一名 AI 助手，能够严格按照给定的上下文生成回答。如果无法直接根据给定的上下文生成回答，仅输出'目前暂无相关信息'"

    # user prompt
    user_prompt = "\n".join(
        [f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
    user_prompt = f"{user_prompt}\nQuestion: {query}"

    # AI response
    ai_response = generate_response(system_prompt, user_prompt)

    # evaluation system prompt
    evaluate_system_prompt = """
    你是一个用于分析AI助手回答的智能评估系统，请按[0, 1]给AI助手的回答打分，分数越高，说明AI助手生成的回答越可靠。
    若AI助手的回答非常符合真实结果，记为1分；
    若AI助手的回答错误、或与真实结果不符，记为0分；
    若AI助手的回答仅有部分符合真实结果，按照真实占比打分（如：0.5）。
    """

    # evaluation prompt = user query + AI response + true response + evaluation system prompt
    evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response}\nTrue Response: {data[0]['answer']}\n{evaluate_system_prompt}"

    # 生成评估结果
    evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)
    print(evaluation_response)
