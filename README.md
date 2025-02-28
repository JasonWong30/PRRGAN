# Journal of Biomedical Health and Informatic 2024

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

Download the Tri-modal image fusion dataset at the [link](https://drive.google.com/drive/folders/1AC_pBejX00iBUKnXWKi73_4Ns7jCtNDZ) and place the sets of images in the folder ``'./Dataset/'``.

The data structure should be followed like:
```
├── Dataset
    ├── Gad-T2-PET
         ├── Y_color
             ├── 1.png
             ├── 2.png
             ├── ...
         ├── other
             ├── 1.png
             ├── 2.png
             ├── ...
         ├── T2
             ├── 1.png
             ├── 2.png
             ├── ...
    ├── CT-T2-SPECT
         ├── Y_color
             ├── 1.png
             ├── 2.png
             ├── ...
         ├── other
             ├── 1.png
             ├── 2.png
             ├── ...
         ├── T2
             ├── 1.png
             ├── 2.png
             ├── ...
```
The user also could modify the ./util/loader.py according to their preferences。

### 🏄 3. Inference

If you want to obtain the dual-modal medical image fusion results in our paper, please run

```
CUDA_VISIBLE_DEVICES=0 python Test_DMIF.py
```

Similarly,  please run

```
CUDA_VISIBLE_DEVICES=0 python Test_v2.py
```

to obtain the Tri-modal medical image fusion results.

The user can modify the argument args.save_dir to locate their own save path.

## 🙌 PRRGAN

### Illustration of our PRRGAN model.

<img src="Fig//Framework.jpg" width="60%" align=center />

### Detail of PRRGAN.

<img src="Fig//GCN.jpg" width="60%" align=center />

<img src="Fig//Share-paramter.jpg" width="60%" align=center />

### Qualitative fusion results.

<img src="Fig//CT-T2-SPECT.jpg" width="60%" align=center />

### Quantitative fusion results.

<img src="Fig//Metrics.jpg" width="60%" align=center />

## 📖 Related Work
- Jingxue Huang, Xiaosong Li, Haishu Ta, Xiaoqi Cheng. Multimodal Medical Image Fusion Based on Multichannel Aggregated Network. The 12th International Conference on Image and Graphics (ICIG), 2023, Nanjing, China. September 22–24. https://link.springer.com/chapter/10.1007/978-3-031-46317-4_2
- Jingxue Huang, Tianshu Tan, Xiaosong Li, Tao Ye, Yanxiong Wu, Multiple Attention Channels Aggregated Network for Multimodal Medical Image Fusion, Medical Physics. https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.17607
