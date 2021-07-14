# Expert augmented deep learning model for prediction of local control in brain metastases
Authors: Taman Upadhaya, Benjamin Ziemer, Tianxiang Zhou, Berkay Canogullari

## Preface
This is the code repository for Expert augmented deep learning model for prediction of local control in brain metastases. Current 3D CNN models rely only on delineated lesions regions or cropped lesions regions of MRIs. However, there is a need to augment the expertise of the clinicians, take into account lesions subsites and also take a holistic view of the whole organ in context within the models. In this work we contributed a deep learning framework exploiting a large dataset of brain metastases to predict local control.

## Summary
In this work, we modified the ResNext architecture from **Aggregated Residual Transformations for Deep Neural Networks<sup>1</sup>**. To further leverage the performance, squeeze-and-excitation(SE) blocks from **Squeeze-and-Excitation Networks<sup>2</sup>** are added to the original architecture. The conv2d layers in ResNext architecture was replaced by conv3d as our input is 3D volume; the last fully connected layer was also replaced by a fully connected layer with output channel equals to 1. Below graphs show a comparison between resnext and resnet (top) and a squeeze-and-excitation(SE) block (bottom).

<img src='https://github.com/naivepig1998/brain_met_3d_cnn/blob/main/images/resnext.png' height=200 width=400>
<img src='https://github.com/naivepig1998/brain_met_3d_cnn/blob/main/images/se.png' height=200 width=400> 


## Code Structure
┣ 📂data    
┃ ┣ 📜meta.csv     
┃ ┗ patient_data    
┗ 📂src    
┃ ┣ 📜loss.py    
┃ ┣ 📜main.py    
┃ ┣ 📜model.py    
┃ ┗ 📜utils.py    

## System Requirements
In order to successfully run this code, you will need to have:
1. GPU(s) with over 11GB of vRAM
2. CPU >= AMD3900
3. RAM >= 32G

## How to run the code
The code is simple to run, please follow the following steps:
1. clone this repository by using `$ git clone`
2. install dependencies   
`$ pip install -r requirements.txt`
3. go to src folder.         
`$ cd src`
4. run the following command
```
 $ python main.py --name test_run \   
                  --image-size 256 \      
                  --debug False \     
                  --init-lr 3e-4 \    
                  --output-dim 1 \    
                  --bs 48 \       
                  --n-epochs 25 \     
                  --num_workers 24 \      
                  --seed 42 \     
                  --percentage 1 \      
                  --use-amp True \      
                  --gpus [0,1,2]
```
