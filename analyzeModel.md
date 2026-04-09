# Role & Context
你现在是一个高级数据科学家和计算机视觉研究员。我正在研究 Video-Text Grounding (视频文本定位) 任务，以 CPL (Contrastive Proposal Learning) 为 baseline，训练了多个模型变体（如 CPL-MoE, CPL-Fusion, CPL-DMVS, CPL-router 等）。

目前我在分析不同模型在 Charades-STA 和 ActivityNet Captions 数据集上的表现差异。由于这两个数据集的样本仅由“视频+自然语言查询+起止时间戳”构成，**缺乏显式的类别标签（如动作类别）**，我需要你帮我编写一套综合评估脚本，通过提取“隐式属性”来对数据集进行切片，并对多个模型的预测结果进行细粒度对比分析。

最终目标是能分析出模型在什么类型的描述文本和视频上拥有较好的性能。

# 输入数据格式
Charades-STA 和 ActivityNet Captions 数据集的相关信息存储在`cpl-main/data`
内容如下：

```json
["AO8RW", 33.67, [0.0, 6.9], "a person is putting a book on a shelf."]
```

`"AO8RW"`表示视频编号，`33.67`表示视频长度，`[0.0, 6.9]`表示该样例定位的实际时序区间，`"a person is putting a book on a shelf."`表示该样例需要被定位的文本。需要注意的是，同一个视频可能会有多个描述文本，他们可能在相近的时序区间，并且也有相似的描述文本，也有可能完全是同一个时序区间只是描述文本有一点差异，也有可能时序区间和描述文本的差异都很大。此外，Charades-STA的视频总长度比ActivityNet Captions小，会有更复杂的情况。

# 数据存储

`cpl-main/data`下有两个文件夹，分别是Charades-STA 和 ActivityNet Captions 数据集的注释文件，`glove.pkl`存储了描述文本的词向量特征。Charades-STA 数据集的视频特征存储在路径`"/data/chenyuan/videogrounding/cpl-main/data/charades/i3d_features.hdf5"`ActivityNet Captions 数据集的视频特征存储在路径`"/data/chenyuan/videogrounding/cpl-main/data/activitynet/sub_activitynet_v1-3.c3d.hdf5"`