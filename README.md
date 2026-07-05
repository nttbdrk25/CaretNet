# Lightweight moment-residual-coherent patterns for image recognition
**Abstract:**
The learning ability of lightweight CNN-based models is usually modest due to lack of
spatial diversity in feature extraction as well as the imperfection of aggregated spatial
information for identity mappings. To deal with these problems, we introduce an efficient
lightweight model by addressing three novel concepts as follows. i) A novel perceptive
block is proposed to extract discriminative moment-residual-coherent features (named
MRCF) from depthwise-based tensors. ii) To adapt to the channel-elasticity moments of
MRCF in a shallow backbone, two novel adaptive residual mechanisms are presented: an
increase-moment residual is based on the expanding flexibility of a pointwise operator,
while the decrease-moment one is on the aggregated spatial patterns of a fused tensor.
To the best of our knowledge, it is the first time that an identify-mapping mechanism is
structured for condensed-spatial information without increasing the model complexity.
iii) A lightweight network is introduced by addressing three robust caret-shape segments
of MRCFs. Experiments on various benchmark datasets have verified the efficacy of our
proposals.

<u>**A sample for training and validating CaretNet on Stanford Dogs:**</u>

```
$ python CaretNet_StanfordDogs.py #for training
$ python CaretNet_StanfordDogs.py --evaluate #for validating
```
**Related citation(s):**

If you use any materials, please cite the following relevant work(s).

```
@article{CaretNetNguyen25,
  author       = {Thanh Tuan Nguyen, Hoang Anh Pham, Thanh Phuong Nguyen, Thinh Vinh Le, Hoai Nam Vu, Van-Dung Hoang},
  title        = {Lightweight moment-residual-coherent patterns for image recognition},
  journal      = {Pattern Recognition Letters},
  volume       = {204}
  page         = {55--63}
  year         = {2026}
}
```
