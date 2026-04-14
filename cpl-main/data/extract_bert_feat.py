import os
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm
from collections import Counter
import nltk
from transformers import BertTokenizer, BertModel

# 确保已经下载了必要的 nltk 数据
# nltk.download('punkt')

def generate_bert_vocab(dataset_name, json_files, output_pkl):
    print(f"\n[Processing {dataset_name}]")
    
    # 1. 初始化 BERT Tokenizer 和 Model
    # 使用 bert-base-uncased (静态词向量维度为 768)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        # 取出静态词向量矩阵后立即释放模型，避免占用 ~440 MB 内存
        bert_embeddings = model.embeddings.word_embeddings.weight.detach().cpu().numpy()
    del model
    bert_vocab = tokenizer.get_vocab()
    
    # 数据结构：w2id 映射词到 ID，id2vec 存储向量，counter 记录词频（用于 base.py 中的词表裁剪）
    vocab = {'w2id': dict(), 'id2vec': [], 'counter': Counter()}
    
    # 2. PAD 占位符 (保持与原项目 vocab.py 逻辑一致)
    vocab['w2id']['<PAD>'] = 0
    vocab['id2vec'].append(np.zeros(768, dtype=np.float32))
    
    # 3. 读取所有 JSON 文件中的句子
    all_sentences = []
    for file_path in json_files:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                # 根据原代码逻辑，句子通常在 index 3
                all_sentences.append(item[3])
                
    # 4. 分词并匹配 BERT 向量
    print(f"Extracting BERT features for {len(all_sentences)} sentences...")
    for sentence in tqdm(all_sentences):
        # 保持与 datasets/base.py 一致，使用 nltk 分词
        tokens = nltk.tokenize.word_tokenize(sentence.lower())
        for word in tokens:
            vocab['counter'][word] += 1
            if word not in vocab['w2id']:
                # 匹配逻辑：
                if word in bert_vocab:
                    # 如果词在 BERT 词典中，直接取向量
                    vec = bert_embeddings[bert_vocab[word]]
                else:
                    # 如果不在（OOV），使用 BERT Tokenizer 处理该词并取子词均值
                    sub_ids = tokenizer.encode(word, add_special_tokens=False)
                    if sub_ids:
                        vec = np.mean(bert_embeddings[sub_ids], axis=0)
                    else:
                        # 极端情况下使用 [UNK]
                        vec = bert_embeddings[tokenizer.unk_token_id]
                
                vocab['w2id'][word] = len(vocab['id2vec'])
                vocab['id2vec'].append(vec.astype(np.float32))
                
    # 5. 保存结果
    with open(output_pkl, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Success! Saved BERT features to {output_pkl} (Dim: 768)")

if __name__ == "__main__":
    # 使用脚本文件所在目录的上一级（即 cpl-main/）作为基准路径，
    # 确保无论从哪个目录执行本脚本，路径均能正确解析
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _p(*parts):
        return os.path.join(_ROOT, *parts)

    # 配置 Charades 数据集路径
    generate_bert_vocab(
        "Charades",
        [_p('data/charades/train.json'), _p('data/charades/test.json')],
        _p('data/charades/bert.pkl')
    )

    # 配置 ActivityNet 数据集路径
    generate_bert_vocab(
        "ActivityNet",
        [_p('data/activitynet/train_data.json'),
         _p('data/activitynet/test_data.json'),
         _p('data/activitynet/val_data.json')],
        _p('data/activitynet/bert.pkl')
    )

'''
# Run from anywhere — paths are now absolute
python cpl-main/data/extract_bert_feat.py
# outputs: data/charades/bert.pkl  and  data/activitynet/bert.pkl

# Then train normally — configs already point to bert.pkl
python cpl-main/train.py --config-path config/activitynet/main_moev2.json
'''