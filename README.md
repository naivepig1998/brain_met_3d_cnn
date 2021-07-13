# Deep Learning Brain Metastasis

## How to run the code
The code is simple to run, please follow the following steps:
1. clone this repository by using `$ git clone`
2. install dependencies   
`$ pip install -r requirements`
3. go to src     
`$ cd src`
4. run the following command
`$ python main.py --name test_run \   
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
                  --gpus [0,1,2]`
