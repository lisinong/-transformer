# 导入必需的库
from torch.utils.data import Dataset
import torch
import jieba
import json


# 定义TextDataset类，该类继承自PyTorch中的Dataset
class TextDataset(Dataset):
    # 初始化函数，filepath为输入文件路径
    def __init__(self, filepath):
        words = []  # 创建一个空列表来存储所有单词

        # 打开文件并读取每一行
        with open(filepath, 'r') as file:
            for line in file:
                # 使用jieba库进行分词，并去除每行的首尾空白字符
                words.extend(list(jieba.cut(line.strip())))

        # 将所有单词转换为一个集合来去除重复，然后再转回列表形式，形成词汇表
        self.vocab = list(set(words))
        self.vocab_size = len(self.vocab)  # 计算词汇表的大小

        # 创建从单词到整数的映射和从整数到单词的映射
        self.word_to_int = {word: i for i, word in enumerate(self.vocab)}
        self.int_to_word = {i: word for i, word in enumerate(self.vocab)}

        # 将映射关系保存为JSON文件
        with open('data/word_to_int.json', 'w') as f:
            json.dump(self.word_to_int, f, ensure_ascii=False, indent=4)
        with open('data/int_to_word.json', 'w') as f:
            json.dump(self.int_to_word, f, ensure_ascii=False, indent=4)

        # 将所有单词转换为对应的整数索引，形成数据列表
        self.data = [self.word_to_int[word] for word in words]

    # 返回数据集的长度减1，这通常是因为在机器学习中可能需要使用当前数据点预测下一个数据点
    def __len__(self):
        return len(self.data) - 1

    # 根据索引idx返回数据，这里用于返回模型训练时的输入序列和目标输出
    def __getitem__(self, idx):
        # 从数据中提取最多50个整数索引作为输入序列
        input_seq = torch.tensor(self.data[max(0, idx - 50):idx], dtype=torch.long)
        # 提取目标输出，即索引位置的单词
        target = torch.tensor(self.data[idx], dtype=torch.long)
        return input_seq, target  # 返回一个元组包含输入序列和目标输出
