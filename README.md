# Pytorch-Zero-Shot-Person-Re-identification-via-Cross-View-Consistency

Implementation of the paper:   
Zero-Shot Person Re-identification via Cross-View Consistency  
Zheng Wang, Ruimin Hu, Chao Liang, Yi Yu, Junjun Jiang, Mang Ye, Jun Chen, and Qingming Leng  

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7346474

[Changes]
Rather chosing from the training samples: 
I'm using it for cross data evaluation, i.e. the training data is replaced by the prototypes from the original dataset (called centers) and query and gallery is taken via another dataset. This is a type of cross-data evaluation. 


## eq2

![eq2](https://github.com/ppriyank/Pytorch-Zero-Shot-Person-Re-identification-via-Cross-View-Consistency/blob/master/eq2.png)


## eq3

![eq3](https://github.com/ppriyank/Pytorch-Zero-Shot-Person-Re-identification-via-Cross-View-Consistency/blob/master/eq3.png)


## eq4

![eq4](https://github.com/ppriyank/Pytorch-Zero-Shot-Person-Re-identification-via-Cross-View-Consistency/blob/master/eq4.png)
