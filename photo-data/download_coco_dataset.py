import json
import os
import requests
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import time

def download_image(url, save_path, max_retries=3):
    """下载图片到指定路径"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                print(f"下载失败 (状态码 {response.status_code}): {url}")
                return False
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"下载出错，重试 {attempt + 1}/{max_retries}: {e}")
                time.sleep(1)
            else:
                print(f"下载失败: {url}, 错误: {e}")
                return False
    return False

def main():
    # 创建图片保存目录
    images_dir = Path("e:/photo-data/images")
    images_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    print("正在加载数据集...")
    dataset = load_dataset("ChristophSchuhmann/MS_COCO_2017_URL_TEXT", split="train", streaming=True)

    # 存储结果
    results = []
    url_to_texts = {}  # 用于聚合同一图片的多个文本

    print("正在处理数据...")
    count = 0

    # 遍历数据集
    for item in tqdm(dataset, total=200, desc="处理数据"):
        if count >= 200:
            break

        url = item['URL']
        text = item['TEXT']

        # 将同一URL的文本聚合在一起
        if url not in url_to_texts:
            url_to_texts[url] = []

        url_to_texts[url].append(text)
        count += 1

    print(f"\n共收集到 {len(url_to_texts)} 个唯一图片")

    # 下载图片并生成JSON数据
    successful_downloads = 0

    for idx, (url, texts) in enumerate(tqdm(url_to_texts.items(), desc="下载图片")):
        # 从URL中提取文件名，或使用索引
        image_filename = f"image_{idx:05d}.jpg"
        image_path = images_dir / image_filename
        relative_path = f"images/{image_filename}"

        # 下载图片
        if download_image(url, image_path):
            # 确保有5个文本描述（如果不足5个，用空字符串填充；如果超过5个，只取前5个）
            while len(texts) < 5:
                texts.append("")
            texts = texts[:5]

            results.append({
                "path": relative_path,
                "text": texts
            })
            successful_downloads += 1
        else:
            print(f"跳过图片 {url}")

    # 保存为JSON文件
    output_file = "e:/photo-data/coco_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完成！")
    print(f"成功下载: {successful_downloads} 张图片")
    print(f"JSON文件已保存到: {output_file}")
    print(f"图片保存在: {images_dir}")

if __name__ == "__main__":
    main()
