

#  PRRGAN

This is official Pytorch implementation of "[Generative Adversarial Network for Trimodal Medical Image Fusion using Primitive Relationship Reasoning].[https://ieeexplore.ieee.org/abstract/document/10620611]"

## To Test TMIF

Run ```**CUDA_VISIBLE_DEVICES=0 python Test_v2.py**``` to test the model. Note that DMIF checkpoint is output_dir/checkpoint-120.pth

## To Test DMIF

Run ```**CUDA_VISIBLE_DEVICES=0 python Test_DMIF.py**``` to test the model. Note that TMIF checkpoint is output_dir/checkpoint-650.pth

If this work is helpful to you, please cite it as:
```
@article{huang2024generative,
  title={Generative Adversarial Network for Trimodal Medical Image Fusion using Primitive Relationship Reasoning},
  author={Huang, Jingxue and Li, Xiaosong and Tan, Haishu and Cheng, Xiaoqi},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}

```

## Recommended Environment

 - [ ] torch  1.12.1
 - [ ] torchvision 0.13.1
 - [ ] numpy 1.24.2
 - [ ] Pillow  8.4.0
