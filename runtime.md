# 运行说明

## 环境

验证环境：Windows、Python 3.11、RTX 5060 8GB、32GB RAM、PyTorch/torchaudio 2.11.0+cu128、FunASR 1.4.16、faster-whisper 1.2.1、CTranslate2 4.8.0。

使用独立 venv。可以在已有 CUDA PyTorch 正常的环境基础上 `python -m venv --system-site-packages .venv`，然后只在新环境安装技能 requirements.txt。新机器的 CUDA PyTorch 按官方对应硬件安装说明准备；PyPI 安装成功不代表 GPU 已可用。验证 `torch.cuda.is_available()`。

模型官方来源：[SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)、[ModelScope](https://modelscope.cn/models/iic/SenseVoiceSmall)。下载后本地目录需包含 model.pt、config.yaml、configuration.json、am.mvn、chn_jpn_yue_eng_ko_spectok.bpe.model。脚本使用本地路径和内置 FunASR 实现，不执行模型仓库远程代码、不上传音频。模型来源和许可以官方说明为准，仓库不分发权重。

本次使用 Hugging Face 配置版本 `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7`。权重从官方 ModelScope 下载，model.pt SHA256：`833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea`。

## 命令

使用所选 venv 的 python。所有路径由用户环境提供；下面是相对当前工作目录的示例。

```sh
ffmpeg -hide_banner -loglevel error -nostdin -n -i original.ts -map 0:a:0 -vn -ar 16000 -ac 1 -c:a pcm_s16le audio.wav
python transcribe.py --audio audio.wav --model models/SenseVoiceSmall --output run-01 --timestamps
```

脚本拒绝覆盖已有输出目录。小样可加 `--start 1800 --duration 960`。默认 BF16、batch=2；FP32 可用 `--dtype float32` 单独测，但不要未经验证就换成 FP16：本次 FP16 样本出现了有语音但空白的窗口。

每批完成后 segments.jsonl 刷新到磁盘，外部消费者可逐行读取。只消费换行结束的完整 JSON，窗口文字包含上下文重叠。result.json 在整次处理成功结束时生成；transcript.txt 是可读的窗口文本，不是逐句字幕。

每行包含 core_start/core_end（去重归属区间）、start/end（输入上下文范围）、text、raw_text，以及启用对齐后的 words/word_timestamps。时间均为相对于 WAV 起点的秒数。若音轨由原片中途提取，必须额外加提取偏移。

可以按字词时间中点归属核心窗口的规则合并重叠词，但应人工检查窗口接缝；不同窗口的识别和对齐并不保证完全一致。

## Agent 阅读与商品整理

已经有同一原片的 SenseVoice JSONL 时直接复用，无需重新跑 ASR：

```sh
python prepare_review.py --segments run-01/segments.jsonl --output reading-01 --chunk-seconds 900 --context-seconds 60
```

此脚本处理完整 JSONL，检查核心窗口连续性与字词时间，按时间中点归属核心窗口，再生成完整文字阅读包；不作语义摘要或自动删口播。`manifest.json` 标出覆盖范围与没有归属字词的窗口，空段需判断是静音/背景音乐还是漏识别。阅读包前后文会重复，汇总时按核心范围与商品身份去重。

Agent 阅读所有目标阅读包后，输出每段的 start/end（原片秒数）、描述性 name、type（正式讲解/返场/预告/提及）、带时间的原文 evidence、uncertainties、clipped_start/clipped_end，以及 review_ranges。输入 WAV 从原片中途提取时需另加已确认偏移。句子或 20 秒窗口的边界不能直接视作产品边界，句内换品须进一步查字词时间和上下文。

对小样先冻结 Agent 独立结果，再与旧索引对照。参考索引不得提前发给负责独立提取的 Agent。样本边缘的截断段只报告可见范围；“首次在样本出现”不等于“首次在整场出现”。

## 可选的第二次 ASR

建立 review-ranges.json：

```json
[{"start": 100, "end": 150, "reason": "相邻款式切换，商品名称不确定"}]
```

```sh
python review_medium.py --audio audio.wav --model models/faster-whisper-medium --ranges review-ranges.json --output review-01
```

复核模型是已有的 CTranslate2 格式 faster-whisper medium，使用本地离线加载。该脚本用 batch=1、beam=1 作第二次识别；冲突仍需原声/画面确认，不能自动认定 medium 正确。默认先由当前模型完整核查文字，根据具体疑点决定是否运行 medium。没有运行时报告未运行，不将文字判断称为听音确认。

## Windows 兼容细节

脚本对 SentencePiece 使用 Python 读取模型字节，避开部分 Windows 原生文件加载器不能处理中文路径的问题。对 SenseVoice 权重使用 `torch.load(mmap=True, weights_only=True)` 严格加载，避免 FunASR 常规加载时复制整份 checkpoint 占满系统提交内存。BF16 用于控制占用与避免本次 FP16 的空白输出。

GPU 显示有空余显存，仍可能因 Windows 系统提交内存不足而分配失败。先记录资源占用并减少任务批量；不要未经用户授权关闭应用或修改分页文件。短段成功后仍需长录播稳定性测试。
