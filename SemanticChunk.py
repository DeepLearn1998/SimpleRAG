import os
import numpy as np
from openai import OpenAI
from main import extract_text_from_pdf
from dotenv import load_dotenv

# 加载 .env
load_dotenv()


class SemanticChunk:
    def get_embedding(self, text, model="doubao-embedding-large-text-240915"):
        client = OpenAI(
            api_key=os.environ.get("api_key"),
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        response = client.embeddings.create(
            model=model,
            input=text,
            encoding_format="float"
        )

        return np.array(response.data[0].embedding)

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def compute_breakpoints(self, similarities, method="percentile", threshold=90):
        # 根据不同的切片方法，计算阈值 threshold
        if method == "percentile":
            threshold_value = np.percentile(similarities, threshold)  # 计算 similarities 90%分位数
        elif method == "standard_deviation":
            mean = np.mean(similarities)
            std_dev = np.std(similarities)
            threshold_value = mean - (threshold * std_dev)
        elif method == "interquartile":
            q1, q3 = np.percentile(similarities, [25, 75])
            threshold_value = q1 - 1.5 * (q3 - q1)
        else:
            raise ValueError("Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")

        return [i for i, sim in enumerate(similarities) if sim < threshold_value]  # 小于 threshold 的索引即为切割点 breakpoints

    def split_into_chunks(self, sentences, breakpoints):
        chunks = []
        start = 0

        for bp in breakpoints:
            chunks.append("。".join(sentences[start:bp + 1]) + "。")
            start = bp + 1

        # 将剩余部分拼接为最后的 chunk
        chunks.append("。".join(sentences[start:]))

        return chunks

    def create_embeddings(self, text_chunks):
        # 生成每个 chunk 对应的 embedding
        return [self.get_embedding(chunk) for chunk in text_chunks]

    def semantic_chunk(self, sentences):
        # 计算 sentences 的对应 embeddings
        embeddings = [self.get_embedding(sentence) for sentence in sentences]
        print(f"Generated {len(embeddings)} sentence embeddings.")

        # 余弦相似度
        similarities = [self.cosine_similarity(embeddings[i], embeddings[i + 1]) for i in
                        range(len(embeddings) - 1)]  # 相邻句子之间计算
        # 按照 90%分位数，寻找 chunk 的切割点
        breakpoints = self.compute_breakpoints(similarities, method="percentile", threshold=90)

        # chunking
        text_chunks = self.split_into_chunks(sentences, breakpoints)
        print(f"Number of semantic chunks: {len(text_chunks)}")  # 展示切片数量
        print(f"First text chunk:\n{text_chunks[0]}")  # (以第一个chunk为例)检查切片结果

        # embedding
        chunk_embeddings = self.create_embeddings(text_chunks)

        return chunk_embeddings


if __name__ == "__main__":
    # 文件路径
    pdf_path = "data/test_document.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    sentences = extracted_text.split("。")  # 按句号(。)分割 text

    sc = SemanticChunk()
    chunk_embeddings = sc.semantic_chunk(sentences)