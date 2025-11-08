# CaretNet: A lightweight model of moment-residual-coherent features for image recognition

**Abstract:**
Structuring a lightweight deep model is one of the essential solutions for real applications in mobile devices or embedded systems.
However, the performance of such networks is usually modest due to the lack of diversity of spatial patterns in feature extraction as well as the imperfection of aggregated spatial information for identity mappings.
To deal with these problems, we introduce an efficient lightweight model by addressing three novel concepts as follows.
i) For the diversity of spatial patterns, a novel perceptron of moment-residual-coherent features (named MRCF) is proposed to make a discriminative representation of moment-residual-coherent patterns that have been extracted and enriched from depthwise-based tensors.
ii) To adapt to the channel-elasticity moments of MRCF in a shallow backbone, two novel adaptive residual mechanisms are presented: an increase-moment residual is based on the expanding flexibility of a pointwise operator, while the decrease-moment one is on the aggregated spatial patterns of a fused tensor.
To the best of our knowledge, it is the first time that an efficient identify-mapping mechanism has been structured to exploit condensed-spatial information without increasing the model complexity.
iii) Finally, a lightweight network is introduced by addressing three robust caret-shape segments of MRCF blocks that allow the learning process to effectively capture the moment-residual-coherent patterns of a given tensor.
Experimental results for image recognition on various benchmark datasets have evidently authenticated the efficiency of our proposals.

<u>**A sample for training and validating CaretNet on Stanford Dogs:**</u>

```
$ python CaretNet_StanfordDogs.py #for training
$ python Train_CaretNet_StanfordDogs.py --evaluate #for validating
```
**Related citation(s):**

If you use any materials, please cite the following relevant work(s).

```
@article{CaretNetNguyen25,
  author       = {Thanh Tuan Nguyen, Hoang Anh Pham, Thanh Phuong Nguyen, Thinh Vinh Le, Hoai Nam Vu, Van-Dung Hoang},
  title        = {CaretNet: A lightweight model of moment-residual-coherent features for image recognition},
  journal      = {ACM Transactions on Multimedia Computing, Communications, and Applications},
  note         = {(submitted in 2025)}
}
```
