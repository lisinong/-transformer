# Minimal Transformer（从零实现）实验

此仓库包含一个简洁的 PyTorch 实现：一个**译码器-编码器的 Transformer模型**，。实现包括：多头自注意力、位置前馈网络（FFN）、残差连接 + LayerNorm、以及正弦位置编码等内容。

## 特性
- 从零实现的模块：Scaled Dot-Product Attention、Multi-Head Self-Attention、FFN、Positional Encoding、Residual + LayerNorm。
- 通过 YAML 配置驱动训练（不依赖外部数据集）。
- 可复现运行、模型检查点与训练曲线输出。

## 快速开始

```bash
# （可选）创建并激活虚拟环境
conda create -n transformer python=3.10 -y
conda activate transformer

# 安装依赖
pip install -r requirements.txt

# 训练（使用 configs/base.yaml）
python train_seq2seq.py --config configs/base.yaml
```

训练完成后，产物会保存到 `runs/<run_name>/`：
- `model.pt`: 训练好的权重
- `config.yaml`: 解析后的配置
- `train_curve.png`: 损失曲线图

## 评估 / 生成

```bash
# 在验证集上评估，并示例生成文本
python eval_seq2seq.py --ckpt runs/exp1/model.pt --prompt "The meaning of life " --steps 200
```

## 文件说明
- `transformer/modules.py` – 注意力、MHA、FFN、位置编码、残差 + LN
- `transformer/seq2seq.py` – 译码器-编码器 Transformer 实现

- `train.py` – 训练循环（AdamW、余弦学习率、梯度裁剪）
- `eval.py` – 评估与文本生成
- `configs/base.yaml` – 超参数配置
- `data/input.txt` – 小型示例语料（可替换为你自己的文本）


