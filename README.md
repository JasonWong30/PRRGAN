# JBHI 2024

Codes for ***Generative Adversarial Network for Trimodal Medical Image Fusion using Primitive Relationship Reasoning. (JBHI 2024)***

[Jingxue Huang](https://github.com/JasonWong30), [Xiaosong Li](https://github.com/lxs6), [Haishu Tan](https://www.fosu.edu.cn/mee/teachers/teachers-external/25647.html), [Xiaoqi Cheng](https://www.fosu.edu.cn/mee/teachers/teachers-jxdzgcx/20469.html)

-[*[Paper]*](https://ieeexplore.ieee.org/abstract/document/10620611)    

## Update
- [2025-1] README.md was modified.
- [2024-6] Codes and config files are public available.

## Citation

```
@article{huang2024generative,
  title={Generative Adversarial Network for Trimodal Medical Image Fusion using Primitive Relationship Reasoning},
  author={Huang, Jingxue and Li, Xiaosong and Tan, Haishu and Cheng, Xiaoqi},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}
```

## Abstract

Medical image fusion has become a hot biomedical image processing technology in recent years. The technology coalesces useful information from different modal medical images onto an informative single fused image to provide reasonable and effective medical assistance. Currently, research has mainly focused on dual-modal medical image fusion, and little attention has been paid on trimodal medical image fusion, which has greater application requirements and clinical significance. For this, the study proposes an end-to-end generative adversarial network for trimodal medical image fusion. Utilizing a multi-scale squeeze and excitation reasoning attention network, the proposed method generates an energy map for each source image, facilitating efficient trimodal medical image fusion under the guidance of an energy ratio fusion strategy. To obtain the global semantic information, we introduced squeeze and excitation reasoning attention blocks and enhanced the global feature by primitive relationship reasoning. Through extensive fusion experiments, we demonstrate that our method yields superior visual results and objective evaluation metric scores compared to state-of-the-art fusion methods. Furthermore, the proposed method also obtained the best accuracy in the glioma segmentation experiment.

### 🌐 Usage

### ⚙ 1. Recommended Environment
```
 - [ ] torch  1.12.1
 - [ ] torchvision 0.13.1
 - [ ] numpy 1.24.2
 - [ ] Pillow  8.4.0
```

### 🏊 2. Data Preparation

Download the Infrared-Visible Fusion (IVF) and Medical Image Fusion (MIF) dataset and place the paired images in the folder ``'./input/'``.

### 🏄 3. Inference

If you want to obtain the dual-modal medical image fusion results in our paper, please run

```
CUDA_VISIBLE_DEVICES=0 python Test_DMIF.py
```

Then, the fused results will be saved in the ``'./output/recon/'`` folder.

Similarly,  please run

```
CUDA_VISIBLE_DEVICES=0 python Test_v2.py
```

to obtain the Tri-modal medical image fusion results.

## 🙌 PRRGAN

### Illustration of our DDFM model.

<img src="image//Workflow1.png" width="60%" align=center />

### Detail of DDFM.


### Qualitative fusion results.


### Quantitative fusion results.

Infrared-Visible Image Fusion

<img src="image//Quantitative_IVF.png" width="100%" align=center />

Medical Image Fusion

<img src="image//Quantitative_MIF.png" width="60%" align=center />

## 📖 Related Work
- Zixiang Zhao, Lilun Deng, Haowen Bai, Yukun Cui, Zhipeng Zhang, Yulun Zhang, Haotong Qin, Dongdong Chen, Jiangshe Zhang, Peng Wang, Luc Van Gool. *Image Fusion via Vision-Language Model.* **ICML 2024**. https://arxiv.org/abs/2402.02235.
- Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang, Kai Zhang, Shuang Xu, Dongdong Chen, Radu Timofte, Luc Van Gool. *Equivariant Multi-Modality Image Fusion.* **CVPR 2024**. https://arxiv.org/abs/2305.11443
