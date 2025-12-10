# GA2-CLIP: Generic Attribute Anchor for Efficient Prompt Tuningin Video-Language Models



> [**GA2-CLIP: Generic Attribute Anchor for Efficient Prompt Tuningin Video-Language Models**](https://arxiv.org/pdf/2511.22125)<br>
> [Bin Wang](https://scholar.google.com/citations?user=Uk43cI4AAAAJ&hl=zh-CN&oi=sra), Ruotong Hu, [Wenqian Wang](https://scholar.google.com/citations?user=3a6qqUYAAAAJ&hl=zh-CN), [Wentong Li](https://scholar.google.com/citations?user=MJjM6BcAAAAJ&hl=zh-CN), [Mingliang Gao](https://scholar.google.com/citations?user=IFEIrUgAAAAJ&hl=zh-CN), [Runmin Cong](https://scholar.google.com/citations?user=-VrKJ0EAAAAJ&hl=zh-CN), [Wei Zhang](https://scholar.google.com/citations?user=qCWuPHsAAAAJ&hl=zh-CN)


[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2511.22125)

Official implementation of the paper "[GA2-CLIP: Generic Attribute Anchor for Efficient Prompt Tuningin Video-Language Models](https://arxiv.org/pdf/2511.22125)".
<hr />

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maple-multi-modal-prompt-learning/prompt-engineering-on-imagenet&#41;]&#40;https://paperswithcode.com/sota/prompt-engineering-on-imagenet?p=maple-multi-modal-prompt-learning&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maple-multi-modal-prompt-learning/prompt-engineering-on-sun397&#41;]&#40;https://paperswithcode.com/sota/prompt-engineering-on-sun397?p=maple-multi-modal-prompt-learning&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maple-multi-modal-prompt-learning/prompt-engineering-on-eurosat&#41;]&#40;https://paperswithcode.com/sota/prompt-engineering-on-eurosat?p=maple-multi-modal-prompt-learning&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maple-multi-modal-prompt-learning/prompt-engineering-on-ucf101&#41;]&#40;https://paperswithcode.com/sota/prompt-engineering-on-ucf101?p=maple-multi-modal-prompt-learning&#41;)

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maple-multi-modal-prompt-learning/prompt-engineering-on-fgvc-aircraft&#41;]&#40;https://paperswithcode.com/sota/prompt-engineering-on-fgvc-aircraft?p=maple-multi-modal-prompt-learning&#41;)

[//]: # ()
[//]: # ()
[//]: # (<hr />)

# :rocket: News

* **(Nov 27, 2025)** 
  * Training and evaluation codes for [GA2-CLIP](https://arxiv.org/pdf/2511.22125), along with pretrained models are released.
  
<hr />

## Highlights

![main figure](docs/main.png)
<p align="justify">  In this work, we propose a video prompt learning methodcalled GA2-CLIP. This method effectively improves thegeneralization from known to unknown categories by introducing hard prompts and generic attribute anchors as abridge. Our innovations are: The proposed video languagefine-tuning generic attribute anchor prompt method cancounteract the semantic narrowing problem in the downstream task; we introduce externally supervised hard andsoft prompts through a nonlinear mapping layer, whichenhances the generalization ability through a competitivelearning mechanism. Extensive experiments validate the effectiveness of the method, and we believe that this workprovides new research directions in the field of videoprompt learning, especially for researchers who lack sufficient experimental conditions. </p>


> **<p align="justify"> Abstract:** *Visual and textual soft prompt tuning can effectively improve the adaptability of Vision-Language Models (VLMs)
> in downstream tasks. However, fine-tuning on video tasks
> impairs the model’s generalization ability to unseen classes.
> Existing methods attempt to mitigate this forgetting effect
> by regularizing the gap between hand-crafted prompts and
> soft prompts, but this also weakens the learning ability
> of soft prompts. To address this challenge, we propose
> a plug-and-play coupling prompt learning framework to
> optimize the generalization performance of V-L models in
> video tasks, with the core motivation of mitigating semantic
> space narrowing during fine-tuning by introducing an externally supervised prompt. Specifically, for textual prompts,
> we introduce pre-trained prompts from other datasets as
> hard prompt tokens. These are concatenated with soft
> prompt tokens and coupled via a learnable mapping layer.
> This competitive prompting approach prevents the semantic space from overfitting to supervised categories. In
> addition, we introduce a set of well-designed irrelevant
> video sets and negative prompts as generic attribute anchors to maintain the generic relevance of the attributes
> in the pre-trained semantic space, thus preserving the generalization ability. Experiments on video tasks demonstrate 
> that our method significantly outperforms state-ofthe-art prompt tuning approaches across generalization
> benchmarks, particularly on base-to-new class prediction.* </p>
> 

# Model Zoo
NOTE: All models in our experiments below uses publicly available ViT/B-16 based CLIP model. The trained model weights against each experiment is provided in tables below.


#### Kinetics-400
| Name  (configs)                                                | Input  | Base Acc. | Novel Acc. |  HM  |                                                                                                                                                                                                                   Model                                                                                                                                                                                                                   |
|----------------------------------------------------------------|:------:|:---------:|:----------:|:----:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [GA2-CLIP](configs/base2novel/finetuning_base2novel/k400)     | 32x224 |   77.0    |    63.3    | 69.5 | [Github](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EVEyFxODEvtFt6FVpuIQvNQBgi5bfxce_nqgzqsjuxB48g?e=rOAu0o)/[log](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EcCHHh5FvnlPnlQTHLUk2v0Bv6MMTWHpkluBiQ1MdbZWFA?e=d3NkTX) |

#### HMDB-51
| Name  (configs)                                                | Input  | Base Acc. | Novel Acc. |  HM  |                                                                                                                                                                                                                   Model                                                                                                                                                                                                                   |
|----------------------------------------------------------------|:------:|:---------:|:----------:|:----:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [GA2-CLIP](configs/base2novel/finetuning_base2novel/hmdb)     | 32x224 |   78.3    |    58.9    | 67.2 | [Github](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ETbI3yeoedBNqvAf3oz-faIBeGDy862_Tx_ZQT1soM6hZQ?e=2Y5Vxg)/[log](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EZ-JcyYOVthCu2pU4ou-AWgBHzMYzWsSKC7eL4KBU3xyLg?e=0bj1ed) |

#### UCF-101
| Name  (configs)                                               | Input  | Base Acc. | Novel Acc. |  HM  |                                                                                                                                                                                                                   Model                                                                                                                                                                                                                   |
|---------------------------------------------------------------|:------:|:---------:|:----------:|:----:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [GA2-CLIP](configs/base2novel/finetuning_base2novel/ucf)     | 32x224 |   96.8    |    75.2    | 84.6 | [Github](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EXwqEdOLKSdIpY6AfTSbRMQB0UqZdTKiaWjw-2gf8Ctcyw?e=h2MvBZ)/[log](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EdOmRlCM4zZJpr-Z497OfB4B5YK8qTiApht1StA7xJ3ClA?e=9zYxfS) |

#### SSv2
| Name  (configs)                                                | Input  | Base Acc. | Novel Acc. |  HM  |                                                                                                                                                                                                                   Model                                                                                                                                                                                                                   |
|----------------------------------------------------------------|:------:|:---------:|:----------:|:----:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [GA2-CLIP](configs/base2novel/finetuning_base2novel/ssv2)     | 32x224 |   18.7    |    14.3    | 16.2 | [Github](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/Ee9-LsJzAeROj0rsXZ_Kq2gBWfDTJX9yI3NhsP3Wx9XT7g?e=QTh28B)/[log](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ETWroKKSa3VJmktA1qGcrUIBSWdSK8JaclCD7GpxXWMMRw?e=bNM8PS) |


### Few-shot results
Below table shows few-shot results of ViFi-CLIP for K=2, 4, 8 and 16.

| Name  (configs)                                                                       | Dataset | K (shots) | Input  | Top-1 Acc. |                                                                    Model                                                                     |
|---------------------------------------------------------------------------------------|:-------:|:---------:|:-------|:----------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|
| [GA2-CLIP](configs/few_shot/finetuning_few_shot/hmdb51/16_32_vifi_clip_2_shot.yaml)  | HMDB-51 |     2     | 32x224 |    61.9    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EZfPCFy69GlLms0xE9hacYsBMRDZolyy5-5kh7urW6U5Hg?e=PRR4dj) |
| [GA2-CLIP](configs/few_shot/finetuning_few_shot/hmdb51/16_32_vifi_clip_4_shot.yaml)  | HMDB-51 |     4     | 32x224 |    64.3    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EYSoKhu-CEdFtDIPDB-9mcYBTocR1z6S4pB2prm8M3y86w?e=MgiPpY) |
| [GA2-CLIP](configs/few_shot/finetuning_few_shot/hmdb51/16_32_vifi_clip_8_shot.yaml)  | HMDB-51 |     8     | 32x224 |    68.1    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EXLoRgDpJERKnxWf6GGGqzoBy-jbAuO-IcV4QSWmtT2mBg?e=piTDRc) |
| [GA2-CLIP](configs/few_shot/finetuning_few_shot/hmdb51/16_32_vifi_clip_16_shot.yaml) | HMDB-51 |    16     | 32x224 |    70.8    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EdA4jgYynRBHrhy1ftn-s9gBFRFYCPdaD5y9AQBClaziWg?e=x2tHpP) |
| [GA2-CLIP](configs/few_shot/finetuning_few_shot/ucf101/16_32_vifi_clip_2_shot.yaml)  | UCF-101 |     2     | 32x224 |    90.0    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ERaxz4xkUBdGkGCKopmsctgBWj0aoxf4eNWRFIQPtZja6A?e=FzpFnl) |
| [GA2-CLIP](configs/few_shot/finetuning_few_shot/ucf101/16_32_vifi_clip_4_shot.yaml)  | UCF-101 |     4     | 32x224 |    91.8    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ETa1Ym63eYtDt9Fzlq_5YuEBcNCPlUPbD12zhc4YGusGyg?e=Z1Si0j) |
| [GA2-CLIP](configs/few_shot/finetuning_few_shot/ucf101/16_32_vifi_clip_8_shot.yaml)  | UCF-101 |     8     | 32x224 |    93.8    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EaHr57kr7GBGno5v6Qb7sLUBERvoInzco0yfbO81davqWQ?e=V2Odqn) |
| [GA2-CLIP](configs/few_shot/finetuning_few_shot/ucf101/16_32_vifi_clip_16_shot.yaml) | UCF-101 |    16     | 32x224 |    95.5    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ERGWnUJHBiVJluMvaUrbDPcB3iIGXAet0W-AfwDJy1bL2w?e=0fSQJb) |
| [GA2-CLIP](configs/few_shot/finetuning_few_shot/ssv2/16_32_vifi_clip_2_shot.yaml)    |  SSv2   |     2     | 32x224 |    6.8     | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EfmVXJyo9VxHheDrVrm7b88BJ_MXRyI_dhuI9pWMUpfPww?e=JPmnt2) |
| [GA2-CLIP](configs/few_shot/finetuning_few_shot/ssv2/16_32_vifi_clip_4_shot.yaml)    |  SSv2   |     4     | 32x224 |    9.9     | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ET1MeS3-C_NLpg-rAJMnf0cBruk16K56NDCwySFwse1tsQ?e=1fV3k2) |
| [GA2-CLIP](configs/few_shot/finetuning_few_shot/ssv2/16_32_vifi_clip_8_shot.yaml)    |  SSv2   |     8     | 32x224 |    10.6     | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EWp7ERV-Dn9GiiTgKWyjDyMBUVoLXyPdHcBpAPah3XvZmw?e=r5Xmii) |
| [GA2-CLIP](configs/few_shot/finetuning_few_shot/ssv2/16_32_vifi_clip_16_shot.yaml)   |  SSv2   |    16     | 32x224 |    14.7    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EZJB66ssj_VBhZB6e59wI9oB1qHGKujTAhoSKyqvnpEzDw?e=Vdjp5n) |


<!--    
# Model Zoo
NOTE: All models in our experiments below uses publicly available ViT/B-16 based CLIP model. The trained model weights against each experiment is provided in tables below.

### Zero-shot results
All models are trained on Kinetics-400 and then evaluated directly on downstream datasets.

| Name  (configs)                                                           | Input  | HMDB-51 | UCF-101 | Kinetics-600 |                                                                    Model                                                                     |
|---------------------------------------------------------------------------|:------:|:-------:|:-------:|:------------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|
| [CLIP image-FT](configs/zero_shot/train/k400/16_16_image_tuned_clip.yaml) | 32x224 |  49.0   |  72.9   |     62.2     | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EdA6n7TCQEFAse5X1g1I08AByLCWHM69axTyK9OyVZy86Q?e=NaipU1) |
| [CLIP text-FT](configs/zero_shot/train/k400/16_16_text_tuned_clip.yaml)   | 32x224 |  48.5   |  69.8   |     68.5     | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/Eea6hW-_RBdJo4T5JJ_sWgEBNGFdA91tPTq9MQ-XkO5dMg?e=hGneeQ) |
| [ViFi-CLIP](configs/zero_shot/train/k400/16_16_vifi_clip.yaml)            | 32x224 |  51.3   |  76.8   |     71.2     | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EW0shb6XYDxFi3BH6DT70rgBPDwgW_knQ8jDsarxINXezw?e=RbixXc) |


### Base-to-novel generalization results
Here, we divide each dataset into base and novel classes.
All models are trained on base classes and evaluated on both base and novel classes. Results are averaged over 3 seeds for each experiment.

#### Kinetics-400
| Name  (configs)                                                | Input  | Base Acc. | Novel Acc. |  HM  |                                                                                                                                                                                                                   Model                                                                                                                                                                                                                   |
|----------------------------------------------------------------|:------:|:---------:|:----------:|:----:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [CLIP image-FT](configs/base2novel/finetuning_base2novel/k400) | 32x224 |   72.9    |    58.0    | 64.6 | [seed1](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EXaQGUrODN9DjxtWuSylHJIBbFtAimZHdubKSHPlTT79eg?e=WiNFM9)/[seed2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ESKX8BXvQoBHn5jq04EoowEB0zR6iPxlkxjSuWJbHupceg?e=vDniMa)/[seed3](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/Ed1D7oXTF6VKtAVSleSJHowBJxsdu1kNNDRk4LBGOfzokg?e=gqt8en) |
| [CLIP text-FT](configs/base2novel/finetuning_base2novel/k400)  | 32x224 |   73.4    |    59.7    | 65.8 | [seed1](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EduVCGSp11tFlwyCKg5ee7wBMJQwGHN9gKNBJozpZCgPEg?e=NPeIjf)/[seed2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ERw9FPot9T9PrVw0kxdsQvkBcpDuDYYUnIFLTjm_xqz8zA?e=8dkLY8)/[seed3](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EU1lFXfDColIuYqGRrujImwBVqz2vP5gpTAM446HPa7erA?e=MCcZ6t) |
| [ViFi-CLIP](configs/base2novel/finetuning_base2novel/k400)     | 32x224 |   76.4    |    61.1    | 67.9 | [seed1](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EVEyFxODEvtFt6FVpuIQvNQBgi5bfxce_nqgzqsjuxB48g?e=rOAu0o)/[seed2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EcCHHh5FvnlPnlQTHLUk2v0Bv6MMTWHpkluBiQ1MdbZWFA?e=d3NkTX)/[seed3](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ETG_gS_l-E1Ai6BkPq8WlzgB8L5PDYDoVrgzia9832j3wg?e=rfJzPs) |

#### HMDB-51
| Name  (configs)                                                | Input  | Base Acc. | Novel Acc. |  HM  |                                                                                                                                                                                                                   Model                                                                                                                                                                                                                   |
|----------------------------------------------------------------|:------:|:---------:|:----------:|:----:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [CLIP image-FT](configs/base2novel/finetuning_base2novel/hmdb) | 32x224 |   62.6    |    47.5    | 54.0 | [seed1](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EVNYYAhsZtZMtzoQcfKx7rQBlrEYkvUyDVfauuMobgAA0g?e=GQ2D8z)/[seed2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EQaX5EzlLfhGhbZEzsgST0cB4HD0saOuoYgBCW7K8bzaBg?e=tKNkqY)/[seed3](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EUiGmlJa3M9Fgx6epCvPiRkBtkON4YKMWEtQSkwqC3dXWw?e=72DTbt) |
| [CLIP text-FT](configs/base2novel/finetuning_base2novel/hmdb)  | 32x224 |   70.0    |    51.2    | 59.1 | [seed1](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ETJ12FfB_8RLg22CHxKPHosBPmFL52G9kbKGayQiqoHXYQ?e=hTb1tv)/[seed2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EXVDioTuv6dKgroWI-qmrEUBZV5njUMUndR_XJDZNTcXcw?e=rgPF49)/[seed3](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EQMykf375n1Iqm2IgABFT2oBokF2ooseZITmvyx2RKX4TA?e=1XNgtI) |
| [ViFi-CLIP](configs/base2novel/finetuning_base2novel/hmdb)     | 32x224 |   73.8    |    53.3    | 61.9 | [seed1](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ETbI3yeoedBNqvAf3oz-faIBeGDy862_Tx_ZQT1soM6hZQ?e=2Y5Vxg)/[seed2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EZ-JcyYOVthCu2pU4ou-AWgBHzMYzWsSKC7eL4KBU3xyLg?e=0bj1ed)/[seed3](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EeUfaRWGtEpPn9hVrpb8pCsBJGAMrGZgXLKOOzNNY1DGqA?e=6B7dJy) |

#### UCF-101
| Name  (configs)                                               | Input  | Base Acc. | Novel Acc. |  HM  |                                                                                                                                                                                                                   Model                                                                                                                                                                                                                   |
|---------------------------------------------------------------|:------:|:---------:|:----------:|:----:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [CLIP image-FT](configs/base2novel/finetuning_base2novel/ucf) | 32x224 |   86.4    |    65.3    | 74.4 | [seed1](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EexxOxwJE8dHtk8ykBn39k4B9OaJK88L-N4c8AYOvj4LNA?e=ZCsYlc)/[seed2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EdC89wvjprhJgG-Q3DuzF_0BoKT0fxQWSeRLgJ6urodhaw?e=U3gU8U)/[seed3](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/Efsw7nYSffZKhzwnIpwX89kBsqSSMhexheB-fb-xFn0fOQ?e=Q69d2d) |
| [CLIP text-FT](configs/base2novel/finetuning_base2novel/ucf)  | 32x224 |   90.9    |    67.4    | 77.4 | [seed1](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EZmAd-E7FXZBoOKa9XY8RfcBG9Qk7nhlLwHin8oN89IKMg?e=Xnmtn9)/[seed2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ERIpU_6ZhUpKjTZ6QQVfKPwBQkUiWLM6yRSOJmFZGOK4-Q?e=pkENDN)/[seed3](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EVoHq04lVOhIpE1pqaI7lmYBhHoh_6Nndgx7xMCZqeXTMw?e=qkcbFm) |
| [ViFi-CLIP](configs/base2novel/finetuning_base2novel/ucf)     | 32x224 |   92.9    |    67.7    | 78.3 | [seed1](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EXwqEdOLKSdIpY6AfTSbRMQB0UqZdTKiaWjw-2gf8Ctcyw?e=h2MvBZ)/[seed2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EdOmRlCM4zZJpr-Z497OfB4B5YK8qTiApht1StA7xJ3ClA?e=9zYxfS)/[seed3](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EdgjBDJ0iXtMpdkNqOE5otcBgWgbfrrQBG1W0wICrD9qiA?e=x7VXl2) |

#### SSv2
| Name  (configs)                                                | Input  | Base Acc. | Novel Acc. |  HM  |                                                                                                                                                                                                                   Model                                                                                                                                                                                                                   |
|----------------------------------------------------------------|:------:|:---------:|:----------:|:----:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [CLIP image-FT](configs/base2novel/finetuning_base2novel/ssv2) | 32x224 |    9.2    |    8.5     | 8.8  | [seed1](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EfLcOFvIHK1Hjj-Yw7z_TQ8BSwmptokbOsPuzWnqAm8iTg?e=3gb20s)/[seed2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EfAL5G3trhJHue-6RTF4HhsBStMma3XEvzWv_0wQnh1YlA?e=sTnbDG)/[seed3](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/Eff55gcBtRxDuCGebyc0zTIBoAPgwDusk0U5jg7-ddjDDg?e=bXB25M) |
| [CLIP text-FT](configs/base2novel/finetuning_base2novel/ssv2)  | 32x224 |   12.4    |    9.5     | 10.8 | [seed1](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EdYLS33jyZZDsIy71Lk3TfwB76xrHL3BIRrUiNeSvWfnWg?e=ndm1JL)/[seed2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EbpzILaqXJBKgPmTKBA32d0BsFrErjRCAwMwaXNKB39G5w?e=FbLCaN)/[seed3](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EY_VHJNKBhlFuir2dL1frOQB5GbG2UeSoG4p65Wh5wOHNg?e=HncWmy) |
| [ViFi-CLIP](configs/base2novel/finetuning_base2novel/ssv2)     | 32x224 |   16.2    |    12.1    | 13.9 | [seed1](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/Ee9-LsJzAeROj0rsXZ_Kq2gBWfDTJX9yI3NhsP3Wx9XT7g?e=QTh28B)/[seed2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ETWroKKSa3VJmktA1qGcrUIBSWdSK8JaclCD7GpxXWMMRw?e=bNM8PS)/[seed3](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/Efl1L1g_OdJHvLu24Yzh3w4BMrTcdll8DilX13lB6rXaFw?e=lLvOiJ) |


#### VL Prompting approach: Base-to-Novel
ViFi-CLIP is first trained on K400 and then vision and language prompts are further fine-tuned on the downstream datasets.

| Dataset (configs)                                       | Input  | Base Acc. | Novel Acc. |  HM  |                                                                                                                                                                                                                   Model                                                                                                                                                                                                                   |
|---------------------------------------------------------|:------:|:---------:|:----------:|:----:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [HMDB-51](configs/base2novel/prompting_base2novel/hmdb) | 32x224 |   77.1    |    54.9    | 64.1 | [seed1](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/Ee1tEk7Tw-dNibQEMVZYBPMBhYj2--lFdIceS1DNN55mUQ?e=qzP1vE)/[seed2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EWxj-A1_EldJggHhBgVTFPIBdcGAXZn1yiWBATvgTKvLYg?e=WLfYUT)/[seed3](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EXT2ezu2RZBEnKuzkEwYb48BE9LYaXoh-cT9dNSruYiKyg?e=b5cbmX) |
| [UCF-101](configs/base2novel/prompting_base2novel/ucf)  | 32x224 |   95.9    |    74.1    | 83.6 | [seed1](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EYNvOOiV0qZIj-YIZlIH-dcBr-8eALRnPse189llN7QiPQ?e=wbbxDB)/[seed2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EeBoMzLQ-YNNtl5YAKS0MmkBoKWpxblQQk3ieT50OtwlQQ?e=16jKbC)/[seed3](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EWwQJkz41o9KgXkgpDoJnjYBCyCD4bV0pBS9XtAD8VpLoQ?e=VKyBNc) |
| [SSv2](configs/base2novel/prompting_base2novel/ssv2)    | 32x224 |   15.8    |    11.5    | 13.3 | [seed1](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ESey1Xo8Ka1HoJtu04xsng0BSTFIRgOty4AwIlnQL7iuJQ?e=n27FNI)/[seed2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EeLJ6F4mXxBHgBj0qQEXkjkBOCImmwSns3J51yG9YIkjAQ?e=eoXWyd)/[seed3](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EQ8Vjdf0t8ZEuJFGlTBwP2sBmDRhM7FWuYmyOh0UZJdhPg?e=ZMppVA) |


### Few-shot results
Below table shows few-shot results of ViFi-CLIP for K=2, 4, 8 and 16.

| Name  (configs)                                                                       | Dataset | K (shots) | Input  | Top-1 Acc. |                                                                    Model                                                                     |
|---------------------------------------------------------------------------------------|:-------:|:---------:|:-------|:----------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|
| [ViFi-CLIP](configs/few_shot/finetuning_few_shot/hmdb51/16_32_vifi_clip_2_shot.yaml)  | HMDB-51 |     2     | 32x224 |    57.2    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EZfPCFy69GlLms0xE9hacYsBMRDZolyy5-5kh7urW6U5Hg?e=PRR4dj) |
| [ViFi-CLIP](configs/few_shot/finetuning_few_shot/hmdb51/16_32_vifi_clip_4_shot.yaml)  | HMDB-51 |     4     | 32x224 |    62.7    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EYSoKhu-CEdFtDIPDB-9mcYBTocR1z6S4pB2prm8M3y86w?e=MgiPpY) |
| [ViFi-CLIP](configs/few_shot/finetuning_few_shot/hmdb51/16_32_vifi_clip_8_shot.yaml)  | HMDB-51 |     8     | 32x224 |    64.5    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EXLoRgDpJERKnxWf6GGGqzoBy-jbAuO-IcV4QSWmtT2mBg?e=piTDRc) |
| [ViFi-CLIP](configs/few_shot/finetuning_few_shot/hmdb51/16_32_vifi_clip_16_shot.yaml) | HMDB-51 |    16     | 32x224 |    66.8    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EdA4jgYynRBHrhy1ftn-s9gBFRFYCPdaD5y9AQBClaziWg?e=x2tHpP) |
| [ViFi-CLIP](configs/few_shot/finetuning_few_shot/ucf101/16_32_vifi_clip_2_shot.yaml)  | UCF-101 |     2     | 32x224 |    80.7    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ERaxz4xkUBdGkGCKopmsctgBWj0aoxf4eNWRFIQPtZja6A?e=FzpFnl) |
| [ViFi-CLIP](configs/few_shot/finetuning_few_shot/ucf101/16_32_vifi_clip_4_shot.yaml)  | UCF-101 |     4     | 32x224 |    85.1    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ETa1Ym63eYtDt9Fzlq_5YuEBcNCPlUPbD12zhc4YGusGyg?e=Z1Si0j) |
| [ViFi-CLIP](configs/few_shot/finetuning_few_shot/ucf101/16_32_vifi_clip_8_shot.yaml)  | UCF-101 |     8     | 32x224 |    90.0    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EaHr57kr7GBGno5v6Qb7sLUBERvoInzco0yfbO81davqWQ?e=V2Odqn) |
| [ViFi-CLIP](configs/few_shot/finetuning_few_shot/ucf101/16_32_vifi_clip_16_shot.yaml) | UCF-101 |    16     | 32x224 |    92.7    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ERGWnUJHBiVJluMvaUrbDPcB3iIGXAet0W-AfwDJy1bL2w?e=0fSQJb) |
| [ViFi-CLIP](configs/few_shot/finetuning_few_shot/ssv2/16_32_vifi_clip_2_shot.yaml)    |  SSv2   |     2     | 32x224 |    6.2     | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EfmVXJyo9VxHheDrVrm7b88BJ_MXRyI_dhuI9pWMUpfPww?e=JPmnt2) |
| [ViFi-CLIP](configs/few_shot/finetuning_few_shot/ssv2/16_32_vifi_clip_4_shot.yaml)    |  SSv2   |     4     | 32x224 |    7.4     | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ET1MeS3-C_NLpg-rAJMnf0cBruk16K56NDCwySFwse1tsQ?e=1fV3k2) |
| [ViFi-CLIP](configs/few_shot/finetuning_few_shot/ssv2/16_32_vifi_clip_8_shot.yaml)    |  SSv2   |     8     | 32x224 |    8.5     | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EWp7ERV-Dn9GiiTgKWyjDyMBUVoLXyPdHcBpAPah3XvZmw?e=r5Xmii) |
| [ViFi-CLIP](configs/few_shot/finetuning_few_shot/ssv2/16_32_vifi_clip_16_shot.yaml)   |  SSv2   |    16     | 32x224 |    12.4    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EZJB66ssj_VBhZB6e59wI9oB1qHGKujTAhoSKyqvnpEzDw?e=Vdjp5n) |

NOTE: Few-shot results for other CLIP Fine-tuned variants are presented in our main paper (Table 3). Model weights for other variants are provided [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/uzair_khattak_mbzuai_ac_ae/Elz1joid4FlAkgDnr_O1ZLMBNxK3jZOlzdAHv5yopYakJQ?e=wyDe8r).

#### VL Prompting approach: Few-shot
ViFi-CLIP is first trained on K400 and then vision and language prompts are further fine-tuned on the downstream datasets in few-shot manner.

| Dataset (configs)                                     | Input  | K=2  | K=4  | K=8  | K=16 |                                                                                                                                                                                                                                                                                      Model                                                                                                                                                                                                                                                                                       |
|-------------------------------------------------------|:------:|:-----|:----:|:----:|:----:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [HMDB-51](configs/few_shot/prompting_few_shot/hmdb51) | 32x224 | 63.0 | 65.1 | 69.6 | 72.0 | [K=2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EQ0oDcnAJLtJt4CmdhnApVIBiWD2YwAO5x01TYy0mpEmzA?e=iFvoSV)/[K=4](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/Ed3LaBQWcrhLgqStigS5HAsBimR0K6DR5l2x_dI6kWuDCA?e=QfYeRd)/[K=8](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EbpdVbtqUUlLt86s2Ze5gBoBAvG4KgWJVbYFVMMErX7Smw?e=lxRFPs)/[K=16](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EQTj4Xe9veRHqmypVCJ6rRMBEMO6Rky3m_V2Q8f7lqrpEw?e=V1DJRH) |
| [UCF-101](configs/few_shot/prompting_few_shot/ucf101) | 32x224 | 91.0 | 93.7 | 95.0 | 96.4 | [K=2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EXz03SRz-NdCmVcWpSt-GEwBxrBWmlGbitXq9iRGz8EczQ?e=zpongw)/[K=4](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EaFyG9bOXUhEnOsviO0BhowBpjvbRcJb9zCehcgyXdhHRQ?e=fl7H6a)/[K=8](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ERM8RDkpandOshsedBwL0fQBrdQd26zjbaBGGGw1XhuTuQ?e=z8GDng)/[K=16](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EbUEylUsTyBOhPZqy99sY8UBMtE0AA46TdY-MTDs8ma0AA?e=g2038u) |
| [SSv2](configs/few_shot/prompting_few_shot/ssv2)      | 32x224 | 6.7  | 7.9  | 10.2 | 13.5 | [K=2](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EbBoLMM3RnNAvoZ3TdoGYSMBFCfsB_gfaz3svxtyKUdxEA?e=KeIl1s)/[K=4](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EbzbcG00RgJFsezk_tnDmQkBCf7wPPIexuKEgUZJKgmMew?e=lEXJ45)/[K=8](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/ESVYfUXIjZ9CppbPt8mgKOABVKSljMNI2JiD9PLkoABSoQ?e=fxI1l1)/[K=16](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/Ec3wlwVsJ4FDprzfChxkZeoBqz4AH7Y4JRF1SjsvMsOWcw?e=ya59zp) |



### Fully-supervised results on Kinetics-400
| Name  (configs)                                                            | FLOPS(G) | Input  | Top-1 Acc. | Top-5 Acc. |                                                                    Model                                                                     |
|----------------------------------------------------------------------------|:--------:|:------:|:----------:|:----------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|
| [CLIP image-FT](configs/fully_supervised/k400/16_16_image_tuned_clip.yaml) |   281    | 16x224 |    82.8    |    96.2    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EdmXN3BQe79BgW1Tuw3Q--QBPbSc4b1N5-ahEIaK-SxRRA?e=e4bLz7) |
| [CLIP text-FT](configs/fully_supervised/k400/16_16_text_tuned_clip.yaml)   |   281    | 16x224 |    73.1    |    91.2    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EeKqDguvX8NPvz5MIKmVPBIBLxL0wkzh0SCmpfs8ZebdZQ?e=2mKBTr) |
| [ViFi-CLIP](configs/fully_supervised/k400/16_16_vifi_clip.yaml)            |   281    | 16x224 |    83.9    |    96.3    | [link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EfqisYTGKlVIiPI0QHG-pxMBuBMA0906jX_kPpaRGw9Ksw?e=TdbaBU) |

## Installation 
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md). 

## Data preparation
Please follow the instructions at [DATASETS.md](docs/DATASETS.md) to prepare all datasets.

# Training 
For all experiments shown in above tables, we provide config files in `configs` folder. For example, to train ViFi-CLIP (tunes both image and text encoder) on Kinetics-400, run the following command:
```
python -m torch.distributed.launch --nproc_per_node=8 \ 
main.py -cfg configs/fully_supervised/k400/16_16_vifi_clip.yaml --output /PATH/TO/OUTPUT 
```

**Note:**
- We recommend keeping the total batch size as mentioned in respective config files. Please use `--accumulation-steps` to maintain the total batch size. Specifically, here the effective total batch size is 8(`GPUs_NUM`) x 4(`TRAIN.BATCH_SIZE`) x 16(`TRAIN.ACCUMULATION_STEPS`) = 512.
- After setting up the datasets as instructed [DATASETS.md](docs/DATASETS.md), only argument in the config file that should be specified is data path. All other settings in config files are pre-set.

For detailed training instructions for all experimental setup, please refer to [TRAIN.md](docs/TRAIN.md).

# Evaluating models
To evaluate a model, please use a suitable config and corresponding model weights. For example, to evaluate ViFi-CLIP with 16 frames on Kinetics-400, run the command below:
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
-cfg configs/fully_supervised/k400/16_16_vifi_clip.yaml --output /PATH/TO/OUTPUT \
--only_test --resume /PATH/TO/CKPT --opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3
``` -->

## Contact
If you have any questions, please create an issue on this repository or contact at dqwangbin@sdut.edu.cn .


# Citation
If you use our approach (code, model or dataset splits) in your research, please consider citing:
```
@article{wang2025ga2,
  title={GA2-CLIP: Generic Attribute Anchor for Efficient Prompt Tuningin Video-Language Models},
  author={Wang, Bin and Hu, Ruotong and Wang, Wenqian and Li, Wentong and Gao, Mingliang and Cong, Runmin and Zhang, Wei},
  journal={arXiv preprint arXiv:2511.22125},
  year={2025}
}
```

# Acknowledgements
Our code is based on [Text4Vis's repository](https://github.com/whwu95/Text4Vis). We sincerely thank the authors for releasing their code. If you use our model and code, please consider citing Text4Vis as well.
