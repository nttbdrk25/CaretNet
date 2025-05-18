# CaretNet: A lightweight model of interleaved spatial features for image recognition

**Abstract:**

* Structuring a lightweight deep model is one of the essential solutions for real
applications in mobile devices or embedded systems. However, the performance
of such networks is usually modest due to the lack of diversity of spatial patterns
in feature extraction as well as the imperfection of aggregated spatial information for identity mappings. To deal with these problems, we introduce an efficient
lightweight model by addressing three novel concepts as follows. i) For the diversity of spatial patterns, a novel perceptron (named BISF) is proposed to make
a discriminative fusion of interleaved spatial features that have been extracted
from depthwise-based tensors. ii) To adapt to the channel-elasticity moments of
BISF in a shallow backbone, two adaptive residual mechanisms are presented:
an increase-moment residual is based on the expanding flexibility of a pointwise
operator, while the decrease-moment one is on the aggregated spatial patterns of
a fused tensor. To the best of our knowledge, it is the first time that an efficient
identify-mapping mechanism has been structured to exploit condensed-spatial
information without increasing the model complexity. iii) Finally, a lightweight
network is introduced by addressing three robust caret-shape segments of BISF blocks that allow the learning process to effectively capture the interleaved spatial
patterns of a given tensor. Experimental results for image recognition on various
benchmark datasets have evidently authenticated the efficiency of our proposals.

<u>**Training and validating CaretNet on Stanford Dogs:**</u>

- For training CaretNet on datasets Stanford Dogs and ImageNet:
```
$ python Train_CaretNet_StanfordDogs.py
$ python Train_CaretNet_ImageNet.py
```
- For validating CaretNet on datasets Stanford Dogs and ImageNet::
```
$ python Train_CaretNet_StanfordDogs.py --evaluate
$ python Train_CaretNet_ImageNet.py --evaluate
```
<u>**Note:**</u>
- Subject to your system, modify these files (*.py) to have the right path to dataset

- For the instance of validation of CaretNet on ImageNet, download its trained model at: [Click here](https://drive.google.com/file/d/106AtFXm9mRM1vf-msBl-XUvvSVU-UZLM/view?usp=drive_link). And then locate the downloaded file as ./checkpoints/ImageNet1k/model_best.pth.tar

**Related citation(s):**

If you use any materials, please cite the following relevant work(s).

```
@article{CaretNetNguyen25,
  author       = {Thanh Tuan Nguyen, Hoang Anh Pham, Thanh Phuong Nguyen, Thinh Vinh Le, Hoai Nam Vu, Van-Dung Hoang},
  title        = {CaretNet: A lightweight model of interleaved spatial features for image recognition},
  journal      = {Machine Learning},
  note         = {(submitted in 2025)}
}
```
