# 直播商品时间戳技能

让 agent 从中文直播带货录播中整理每个商品的讲解起止时间。使用本地 SenseVoiceSmall 快速转写和时间对齐，再对疑点用已有 Whisper medium、原声或画面复核。

技能入口：[SKILL.md](SKILL.md)。[环境与运行方式](runtime.md)。[实测和限制](validation.md)。

## 安装到 Codex

将本仓库内容放到用户的 `~/.codex/skills/live-product-timestamps/`，确保该文件夹中直接包含 `SKILL.md`，然后在新对话中使用 `$live-product-timestamps`。模型与 Python 依赖另行准备；复制技能本身不会下载模型。

示例请求：

> 使用 $live-product-timestamps 分析这场原始直播录播，先给我商品讲解时间段，不剪片。没有商品清单，按录播识别，不确定的名称标出来。

支持已有音频分段持续写出，便于提前查看结果；当前不含直播拉流或原生逐字流式识别。

仓库仅包含通用技能、脚本和汇总验证记录，不包含录播、转写、截图、凭证或模型权重。商品名与切换边界需要语义判断，ASR 的窗口边界不能直接当作剪辑点。
