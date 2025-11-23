# WV-MOS 指标支持

## 简介

WV-MOS 是基于 wav2vec2.0 微调的 MOS（Mean Opinion Score）预测模型，用于非侵入式语音质量评估。

- **GitHub**: https://github.com/AndreevP/wvmos
- **论文**: https://arxiv.org/abs/2203.13086

## 安装

```bash
pip install git+https://github.com/AndreevP/wvmos
```

## 使用方法

### 1. 通过 MetricRegister 使用

```python
from RSB.evaluate import MetricRegister

# 获取 wvmos 指标
metric_fns = MetricRegister.fetch(['wvmos'])
wvmos_metric = metric_fns['wvmos']()

# 计算指标
result = wvmos_metric.calculate(
    deg_wav=enhanced_audio,  # 增强后的音频（numpy array）
    wav_path="path/to/audio.wav",  # 或者直接传文件路径
    sample_rate=16000
)
print(result)  # {'WV_MOS': 3.85}
```

### 2. 通过命令行使用

```bash
python calc_metrics.py \
    --enhanced_dir /path/to/enhanced/audio \
    --metrics wvmos dnsmos utmos \
    --output_dir ./results
```

### 3. 与其他指标组合使用

```python
from RSB.evaluate import MetricRegister

# 获取多个指标
metric_fns = MetricRegister.fetch(['wvmos', 'dnsmos', 'utmos', 'sigmos'])

# 初始化
for name in metric_fns:
    metric_fns[name] = metric_fns[name]()

# 计算所有指标
results = {}
for name, metric in metric_fns.items():
    results.update(metric.calculate(
        deg_wav=enhanced_audio,
        wav_path="path/to/audio.wav",
        sample_rate=16000
    ))

print(results)
# {'WV_MOS': 3.85, 'DNSMOS_P808': 3.2, 'UTMOS': 3.5, 'SIGMOS_OVRL': 3.7}
```

## 输出格式

WV-MOS 指标输出单个值：

```python
{
    "WV_MOS": float  # 范围通常在 1-5 之间
}
```

## 注意事项

1. WV-MOS 需要 GPU 支持（CUDA），如果不可用会自动回退到 CPU
2. 首次使用时会自动下载预训练模型
3. WV-MOS 是非侵入式指标，只需要增强后的音频，不需要参考音频

## 相关论文

```bibtex
@article{andreev2022wvmos,
  title={WV-MOS: MOS score prediction by fine-tuned wav2vec2.0 model},
  author={Andreev, Pavel and Patakin, Nikolay and Desheulin, Oleg and Kagan, Alexander and Bulanbaev, Arthur},
  journal={arXiv preprint arXiv:2203.13086},
  year={2022}
}
```
