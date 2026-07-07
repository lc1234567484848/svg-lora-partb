# Part B SVG Logo LoRA 作业说明

本仓库是大模型课程 Part B 作业提交项目。任务目标是使用 `ms-swift` 对本地 `Gemma 3 270M` 基座模型做 LoRA SFT，使模型从文字描述生成单个 SVG 徽标，并用自定义 `reward.py` 对生成结果做自评。

最终实验中，`learning_rate=1e-3` 的 V3 版本效果最好：验证集 17 条样本中有 5 条生成了有效 SVG，平均 reward 为 `0.27888`，有效 SVG 比例为 `0.29412`。详细训练日志、四轮学习率对比和失败分析见 `report.md`。

## 目录结构

```text
.
├── adapter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── student_kit/
│   ├── __init__.py
│   ├── eval_self.py
│   ├── eval_self_vllm.py
│   ├── prepare_adapter.py
│   └── utils.py
├── reward.py
├── train_config.yaml
├── results.json
├── report.md
├── README.md
└── .gitignore
```

未提交到 GitHub 的本地目录：

- `pretrained_model/`：本地基座模型目录，体积较大，不上传。
- `output/`：训练过程 checkpoint 和日志目录，体积较大，不上传；关键日志已整理进 `report.md`。

## 文件说明

| 文件或目录 | 作用 |
| --- | --- |
| `reward.py` | 自定义 SVG reward。提供 `score_svg(prompt, svg)` 和 `evaluate_svg(prompt, svg)` 两个接口，用于判断 SVG 是否结构有效、安全、图元合理、坐标合理、颜色合理，并做弱语义匹配。 |
| `train_config.yaml` | `ms-swift` SFT 训练配置。最终使用 `learning_rate: 1e-3`、LoRA rank 8、`max_length: 3072`。 |
| `adapter/` | 最终整理后的 LoRA adapter，包含 `adapter_config.json` 和 `adapter_model.safetensors`。 |
| `results.json` | V3 adapter 在验证集上的自评结果，包含 base 与 LoRA 的逐样本输出、reward 分项、均值和 delta。 |
| `report.md` | 中文实验报告，包含 reward 设计、训练配置、V0-V3 学习率对比、完整 `logging.jsonl`、评测结果、失败分析和 Goodhart 风险分析。 |
| `student_kit/prepare_adapter.py` | 从 `output/` 中寻找 best checkpoint，并复制整理成根目录 `adapter/`。 |
| `student_kit/eval_self_vllm.py` | vLLM 版本评测脚本，速度较快，推荐用于完整验证集评测。 |
| `student_kit/eval_self.py` | Transformers/Peft 版本评测脚本，速度较慢，作为 vLLM 不可用时的备用方案。 |
| `student_kit/utils.py` | JSONL 读取、prompt 构造、SVG 清洗和 reward 汇总工具函数。 |

## 环境准备

本实验在windows、RTX5090、 `lichun` conda 环境中完成，主要依赖包括：

- `ms-swift`
- `torch`
- `transformers`
- `peft`
- `datasets`
- `accelerate`
- `vllm`

进入环境：

```powershell
conda activate lichun
```

确认本地模型和数据存在：

```text
pretrained_model/gemma-3-270m
data/train.jsonl
data/valid.jsonl
```

数据格式为 JSONL，每行包含 `messages` 字段，结构类似：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<svg ...>...</svg>"}
  ]
}
```

## 训练复现

运行 SFT：

```powershell
conda run -n lichun swift sft train_config.yaml
```

当前 `train_config.yaml` 对应最终 V3 配置，核心参数如下：

```yaml
model: pretrained_model/gemma-3-270m
template: gemma3_text
tuner_type: lora
target_modules: all-linear
lora_rank: 8
lora_alpha: 32
learning_rate: 1e-3
num_train_epochs: 5
max_length: 3072
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
eval_steps: 1
save_steps: 1
```

训练输出默认写入：

```text
output/gemma-3-270m/logo_svg_lora
```

本次报告中实际对比了四个学习率版本：

| 版本 | learning rate | 结果概述 |
| --- | ---: | --- |
| V0 | `1e-5` | 0/17 有效 |
| V1 | `1e-4` | 1/17 有效，命中第 3 个样本 |
| V2 | `5e-4` | 1/17 有效，命中第 16 个样本 |
| V3 | `1e-3` | 5/17 有效，当前最佳 |

## 整理 Adapter

训练完成后，将 best checkpoint 整理为根目录 `adapter/`：

```powershell
conda run -n lichun python student_kit/prepare_adapter.py --output-dir output/gemma-3-270m/logo_svg_lora --adapter-dir adapter
```

如果只想从某个具体版本目录整理，例如 V3：

```powershell
conda run -n lichun python student_kit/prepare_adapter.py --output-dir output/gemma-3-270m/logo_svg_lora/v3-20260707-193501 --adapter-dir adapter
```

整理后应得到：

```text
adapter/adapter_config.json
adapter/adapter_model.safetensors
```

## 评测复现

推荐使用 vLLM 评测：

```powershell
conda run -n lichun python student_kit/eval_self_vllm.py data/valid.jsonl --base-model pretrained_model/gemma-3-270m --adapter adapter --out results.json --max-new-tokens 3072 --batch-size 8
```

脚本会评测两组模型：

1. base model：`pretrained_model/gemma-3-270m`
2. LoRA adapter：`adapter/`

输出写入：

```text
results.json
```

`results.json` 中包含：

- `base.summary`
- `base.items`
- `finetuned.summary`
- `finetuned.items`
- `delta`

其中每条样本包含：

- prompt
- raw output
- 清洗后的 SVG
- reward 总分
- reward 分项
- 是否有效 SVG

如果 vLLM 不可用，可以使用 Transformers/Peft 备用脚本，但速度会慢很多：

```powershell
conda run -n lichun python student_kit/eval_self.py data/valid.jsonl --base-model pretrained_model/gemma-3-270m --adapter adapter --out results.json --max-new-tokens 3072
```

## Reward 说明

`reward.py` 的主要接口：

```python
from reward import score_svg, evaluate_svg

score = score_svg(prompt, svg)
detail = evaluate_svg(prompt, svg)
```

reward 检查内容包括：

- 是否是单个 `<svg>...</svg>` 根元素
- XML 是否可解析
- 是否包含 `viewBox="0 0 256 256"`
- 是否禁止 `<script>`、`image`、`foreignObject` 等不安全或外部依赖元素
- 是否包含合理数量的 SVG 图元
- 坐标是否大致落在 256x256 画布内
- 颜色数量是否合理
- 输出长度是否正常
- prompt 中颜色词、形状词与 SVG 内容是否有弱匹配

注意：reward 是自评指标，不等价于人工审美评价。它更偏向判断 SVG 是否结构有效、安全、可解析。

## 最终结果摘要

当前提交的 `results.json` 对应 V3：

| 模型 | 平均 reward | 有效 SVG 比例 |
| --- | ---: | ---: |
| Base Gemma 3 270M | 0.00000 | 0.00000 |
| LoRA V3 | 0.27888 | 0.29412 |
| Delta | +0.27888 | +0.29412 |

V3 有效样本：

```text
idx = 0,  score = 0.965000
idx = 6,  score = 0.965000
idx = 9,  score = 0.904333
idx = 11, score = 0.953333
idx = 12, score = 0.953333
```

详细分析见 `report.md`。

## GitHub 提交说明

本仓库建议提交：

```text
README.md
reward.py
train_config.yaml
results.json
report.md
student_kit/
adapter/
.gitignore
```

不建议提交：

```text
pretrained_model/
output/
__pycache__/
```

原因是模型和训练 checkpoint 体积较大，不适合直接放入普通 GitHub 仓库。关键训练日志已经嵌入 `report.md`。
