# AlternusVera: User-Based Factor
ProjectDeadass  
Allen Wu 015292667 yanshiun.wu@sjsu.edu

[Demo Video](https://drive.google.com/file/d/11cvwsm8JXFGxWKDG4Mvso2T4lRpEtfKG/view?usp=sharing)

[Colab](https://drive.google.com/file/d/16zwCR5jlhYgkRAzNq8iTOrgeM_Mz4BDZ/view?usp=sharing)

[Paper](https://drive.google.com/file/d/1hM0AAZXkhuShQ6XoCnUyQlF__ZfSdmo_/view?usp=sharing)

## Split model2.pkl
split -b90M -d model2.pkl model2.pkl.part_

## Restore model2.pkl
cat model2.pkl.part_* > model2.pkl
