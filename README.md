# COLORGAN
This is codes for Unsupervised Diverse Colorization via Generative Adversarial Networks

## Requirement
tensorflow: 1.0+  
numpy  
scipy  
Pillow  
cv2  

## Usage
####Prepare data  
Download [LSUN/bedroom_train](http://lsun.cs.princeton.edu) data.  
Maximum center crop the data and reshape into 64*64,  
put reshaped data into data/lsun_64/bedroom_train/
####Train  
run train_lsun_wgan.sh for shortcut  
or adjust settings by:
```
python main_wgan.py --is_train=True --some_paramter=some_value
```
####Test
run test_lsun_wgan.sh for shortcut  
or adjust settings by:
```
python main_wgan.py --is_train=False --some_paramter=some_value
```
####Demo  
Some demo results in demo/gan/

