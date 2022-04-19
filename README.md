<h1 align="center">Sartorius Kaggle Experimentations</h1>
<p align="center">
<img src="https://i.imgur.com/B9t4uIj.jpg">
</p>

- It was a cv + medical competition, where we given RLE data for different neuronal cell [shsy5y, cort and astro] segmentation. For details explanation of the data and cell types you can check out my NB [here](https://www.kaggle.com/soumya9977/residual-unet-with-attention-eda-tta-w-b). 
- It was my first competition on kaggle. I previously did one melanoma classification in my 2nd year but it was very overwhelming for me. So I resigned early. But this time I was there will the last hour of the competition.
- I was not able to get any medal in this competition, but luckily we were able to get into top 15% of the LB. It also helped me understand how to approach a kaggle problem. Except that another good thing that happened was one of my [teammate](https://github.com/r-matsuzaka) became Notebook and Discussion Expert.
- I started 4 days late, at first I tried to understand the data, evaluation matric etc. I started with creating a [Unet based model with Attention and residual connection](https://www.kaggle.com/soumya9977/residual-unet-with-attention-eda-tta-w-b), but it did not really give any good results although the inference was looking very neat. I then moved on to Pretrained Unet model with different backbones [resnet,efficientnet-b2]. Still it was not good enough [0.15+]. 
- When I was doing these experiments people on kaggle converged on a best performing single model, which was Mask RCNN. So I started experimenting with Mask RCNN. We did not have any clue about what is cross validation what are different pre and post processing techniques we can use etc. But one of my teammates [r-matsuzaka](https://github.com/r-matsuzaka) was researching on different methods to use. As per his advice I trained MS RCNN [which suppoed to outperform MRCNN] It was giving good results but not as good as the high performing MRCNN models[performances shared in the forums].
- Another breakthrough came to the picture when [Slawek Biel](https://www.kaggle.com/slawekbiel) shared his NB on cellpose. It outperformed MRCNN. We started to shift in cellpose. after 4-5 days of grinding with cellpose we came to a conclusion that the provided github repo for cellpose does not work. 
- As it was already near deadline I started doing Ensemble. I had no clue how to do that, but people already shared some NBs on ensemble based on NMS, NMW. I hacked some of the code to make it work for my models. At the end I used some of my models and some of high scoring public models to do the ensumble. We were able so submit only 52 submissions. It was because of some of the frameworks were new to us, we got distracted with other competitions, less experience, very less experiment tracking etc.    

## Inference image:
<pre>
<img src="https://i.imgur.com/oHPFaMh.png" width="900"> <img src="https://i.imgur.com/xVDB8SO.png" width="900"> <img src="https://i.imgur.com/DN8BvHv.png" width="900"> <img src="https://i.imgur.com/q51Aq6N.jpg" width="900">
</pre>

## Few plots:
<pre>
<img src="https://i.imgur.com/cffhTjc.png" width="900"> <img src="https://i.imgur.com/YPtbx7o.png" width="900"> <img src="https://i.imgur.com/CsMp6bH.png" width="900"> 
</pre>

## Important Kaggle Discussions:
1. [Instance Segmentation Models](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/278883 ) 
2. [Top GitHub Source Codes on Cell Instance Segmentation](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/280172)
3. [Previous Competitions on Instance Segmentation](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/278735)
4. [Relevant past Comps and its Top solutions + extra](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/discussion/280291)


## Plans:
1. -[x] attention base UNET
2. -[x] MRCNN using detectron2 / or using pytorch 
3. -[ ] semantic segmentation using ResNet + Xgboost
4. -[ ] Experiment with the weights of MS RCNN
5. -[x] Experimenting with previously downloaded model files
6. -[x] **Perform Ensemble with different models**
7. -[ ] Use Detectron2 for MS RCNN [might not be a good idea]
8. -[x] **Add TTA with MS RCNN**
9. -[ ] **Use modified data for trianing.**
10. -[ ] **train PointRend with Detectron2.**
11. -[x] **Use MRCNN with ResNet101 Detectron2.**
   11. -[x] **Do inference on MRCNN Detectron2.**
12. -[x] **Train cellpose on cyto2 with high epoch,**
   13. -[x] **Do inference on cellpose.**
14. -[x] **increase the data using augmentation.**
15. -[x] try cellpose
16. -[ ] Use this [cellpose](https://github.com/Gladiator07/Sartorius-Neuronal-Cell-Segmentation-Kaggle) for doing the cellpose training. **Modified cellpose**.

### ways to solve the problem:
- change the threshold
- right now(only 6th epoch) its showing -ve corelation with LB. checking the 5th, if that turns out the same we need to assume the same that the cv is -ve corelated with LB. and check by submitting one.
- I will make the MS RCNN(with high epoch), PointRend, Cellpose in training and then start working on what other combination, we can use that for ensambling and TTA.
- Go back to MSRCNN/CASCADE MSRCNN with folds.
- hyperparameter tune the model config

## Baseline Model & Score Log:
|Model/changes|NB version|LB|standing|
|---|------|--------|-----------------|
|AttentionResUNet|Notebook Attention based Residual Unet + EDA on cell imgs🧬 (Version 25)|`0.036 LB` | `622th`|
|AttentionResUNet with TTA |Notebook checking out the Overlapping problem ( Version 5)|`0,053 LB` | `708th`|
|MMDetection- MS RCNN|[MMDetection Neuron Inference](https://www.kaggle.com/soumya9977/mmdetection-neuron-inference) (V1) [5th final epoch]|`0.233`|`NA`|
|Detectron2 MRCNN|[Inference and Submission fb3065](https://www.kaggle.com/soumya9977/inference-and-submission-fb3065?scriptVersionId=82753213) (v3) [5th epoch]|`0.296`|`NA`|
|Detectron2 MRCNN|[Inference and Submission](https://www.kaggle.com/soumya9977/inference-and-submission?scriptVersionId=82683243) (v1) [model_best_7]|`0.305`|`200`|

## MS RCNN Score Board:

|epoch|threshold|LB|CV|
|-----|---------|--|--|
|6th_epoch|False|`0.231`|`0.2777098`|
|6th_epoch|True|`0.285`|`0.265784`|
|5th_epoch|False|` 0.233`|`0.277636`|
|5th_epoch|True|`0.285`|`0.26764786`|

The table shows that the cv and lb is corelated +ve ly, coz if you see the threshold =True for both the epochs, then the LB is increasing than cv.

## Submissions:
- ### [`Residual Unet with Attention+ EDA +TTA + W&B🧬`](https://www.kaggle.com/code/soumya9977/residual-unet-with-attention-eda-tta-w-b) `80+ Upvotes` 
| NB                                                                                                                                                                   | private | public |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ------ |
| [Ensemble NMS - Detectron2 [Inference]](https://www.kaggle.com/soumya9977/ensemble-nms-detectron2-inference?scriptVersionId=83969029)                                | 0.312   | 0.305  |
| [soEnsemble Non Maximum Weighed- Detectron2 [Infer] -v3](https://www.kaggle.com/soumya9977/soensemble-non-maximum-weighed-detectron2-infer?scriptVersionId=83887547) | 0.313   | 0.306  |
| [soEnsemble Non Maximum Weighed- Detectron2 [Infer] -v2](https://www.kaggle.com/soumya9977/soensemble-non-maximum-weighed-detectron2-infer?scriptVersionId=83871516) | 0.311   | 0.303  |
| [soEnsemble Non Maximum Weighed- Detectron2 [Infer] -v1](https://www.kaggle.com/soumya9977/soensemble-non-maximum-weighed-detectron2-infer?scriptVersionId=83864362) | 0.310   | 0.301  |
| [somu: Detectron2 MRCNN[inference]](https://www.kaggle.com/soumya9977/somu-detectron2-mrcnn-inference?scriptVersionId=83815636)                                      | 0.172   | 0.171  |
| [MMDetection Neuron Inference-> wo_thres-15th](https://www.kaggle.com/soumya9977/mmdetection-neuron-inference-wo-thres-15th?scriptVersionId=83804425)                | 0.178   | 0.173  |
| [somu: Detectron2 MRCNN[inference]](https://www.kaggle.com/soumya9977/somu-detectron2-mrcnn-inference?scriptVersionId=83378703)                                      | 0.265   | 0.265  |
| [# MMDetection Neuron Inference-> wo_thres-15th -v9](https://www.kaggle.com/soumya9977/mmdetection-neuron-inference-wo-thres-15th?scriptVersionId=83199281)          | 0.239   | 0.231  |
| [# MMDetection Neuron Inference-> wo_thres-15th -v7](https://www.kaggle.com/soumya9977/mmdetection-neuron-inference-wo-thres-15th?scriptVersionId=83166396)          | 0.295   | 0.285  |
| [# Inference and Submission 30a1d1](https://www.kaggle.com/soumya9977/inference-and-submission-30a1d1?scriptVersionId=83009549)                                      | 0.312   | 0.306  |
| [# MMDetection Neuron Inference-> wo_thres-15th  -v2](https://www.kaggle.com/soumya9977/mmdetection-neuron-inference-wo-thres-15th?scriptVersionId=82923485)         | 0.296   | 0.285  |
| [# Inference and Submission fb3065 -v6](https://www.kaggle.com/soumya9977/inference-and-submission-fb3065?scriptVersionId=82842606)                                  | 0.310   | 0.303  |
| [# Inference and Submission fb3065 -v5](https://www.kaggle.com/soumya9977/inference-and-submission-fb3065?scriptVersionId=82799129)                                  | 0.313   | 0.303  |
| [# MMDetection Neuron Inference-> wo_thres-15th -v1](https://www.kaggle.com/soumya9977/mmdetection-neuron-inference-wo-thres-15th?scriptVersionId=82773104)          | 0.239   | 0.233  |
| [# Inference and Submission fb3065 -v3](https://www.kaggle.com/soumya9977/inference-and-submission-fb3065?scriptVersionId=82753213)                                  | 0.303   | 0.296  |
| [# Inference and Submission fb3065 -v2](https://www.kaggle.com/soumya9977/inference-and-submission-fb3065?scriptVersionId=82746017)                                  | 0.293   | 0.284  |
| [Inference and Submission fb3065](https://www.kaggle.com/soumya9977/inference-and-submission-fb3065?scriptVersionId=82743942)                                        | 0.293   | 0.284  |
| [Sartorius transfer learning [inference]](https://www.kaggle.com/osamurai/sartorius-transfer-learning-inference?scriptVersionId=82706826)                            | 0.306   | 0.300  |
| [Inference and Submission](https://www.kaggle.com/soumya9977/inference-and-submission?scriptVersionId=82683243)                                                      | 0.312   | 0.305  |
| [Sartorius transfer learning [inference]](https://www.kaggle.com/soumya9977/sartorius-transfer-learning-inference?scriptVersionId=82216003) |0.306         |0.300        |


# **Models Trained** - (with important links)
- [CellPose](https://github.com/danielbarco/malatec/blob/main/Notebooks/cellpose_run.ipynb)
- [PointRend - Detectron2 Colab Approach](https://colab.research.google.com/drive/1J0aNSc63s5aLMeTgW0qVXYFOcTwLbU03?usp=sharing) and [Basic Tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=tjbUIhSxUdm_)
  - [Detectron2 Github](https://github.com/facebookresearch/detectron2)
  - [Detectron2 Docs](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)
  - Custom Datasets [Medium](https://medium.com/@chengweizhang2012/how-to-train-detectron2-with-custom-coco-datasets-4d5170c9f389) and [Diology](https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-instance-segmentation/)
  - [Github Issues - How to Train Pointrend Detectron2 on Custom Dataset](https://github.com/facebookresearch/detectron2/issues/1017)


<!-- >> - skimage.segmentation.relabel_sequential ?
>> - np.pad() ?
 -->
