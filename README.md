# AlternusVera: User-Based Factor
ProjectDeadass  
Allen Wu 015292667 yanshiun.wu@sjsu.edu

## Split model2.pkl
split -b90M -d model2.pkl model2.pkl.part_

## Restore model2.pkl
cat model2.pkl.part_* > model2.pkl
