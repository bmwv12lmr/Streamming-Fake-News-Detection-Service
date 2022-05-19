# AlternusVera: User-Based Factor
ProjectDeadass  
Allen Wu 015292667 yanshiun.wu@sjsu.edu

[Demo Video](https://drive.google.com/file/d/11cvwsm8JXFGxWKDG4Mvso2T4lRpEtfKG/view?usp=sharing)

[Train Video](https://drive.google.com/file/d/1W6A3exxuCQMHAFQTB0IEdGZz4-VJ_II9/view?usp=sharing)

[Colab](https://drive.google.com/file/d/16zwCR5jlhYgkRAzNq8iTOrgeM_Mz4BDZ/view?usp=sharing)

[Paper](https://drive.google.com/file/d/1hM0AAZXkhuShQ6XoCnUyQlF__ZfSdmo_/view?usp=sharing)

## Introduction
This Project is using user-based factor to do streamming Fake News Detection.

## Dataset
1. Liar Liar Dataset
2. Large Movie Review Dataset
3. Genuine/Fake User Profile Dataset

## Instruction
**Training Stage:**  
Use Colab to train the models using three datasets

**Deploy Stage:**  
Download the models and using Google news RSS feed API for streamming latest news from CNN and FOX NEWS.

## Models
1. model1.pkl: Twitter Username Based
2. model2.pkl: Comments Emotional Based
3. model3.pkl: Fake History Based

## Matrix Table
TruePositive= 37  
TrueNegative= 18  
FalsePositive= 38  
FalseNegative= 7  
Precision= 0.49  
Recall= 0.84  
Accuracy= 0.55  

## Installation Notes
Due to the size of model2.pkl is larger than github 100MB limitation, it must be splitted into few parts to upload.
Restore it before use.

**Split model2.pkl**  
split -b90M -d model2.pkl model2.pkl.part_  

**Restore model2.pkl**  
cat model2.pkl.part_* > model2.pkl  
