# ECE253Project
## Denoising
bm3d code:
bm3d_batch.py --images ../../data/noisy/preprocessed --noise-std 30 --save ../../data/noisy/bm3d_out

swinIR is used exactly as their readme says, only redirected output to run different experiments for our work 

## Dehazing
### Traditional Method (Pixel-wise alpha blending)
find the scripts for the code under: dehazing/nighttime_dehaze_traditional.
This dehazing algorithm was adopted from the paper: [T. Yu, K. Song, P. Miao, G. Yang, H. Yang and C. Chen, "Nighttime Single Image Dehazing via Pixel-Wise Alpha Blending," in IEEE Access, vol. 7, pp. 114619-114630, 2019.](https://ieeexplore.ieee.org/document/8805086).

We applied some of the image preprocessing steps before we actually run the code in the matlab. You can find the preprocessing steps inside the dehazing/nighttime_dehaze_traditional/improved_nighttime_dehaze.py. After we run the preprocessing steps on our dataset we need to go to the matlab and run the dehazing/nighttime_dehaze_traditional/nt_dehaze.p.  
Since the matlab code was private code we could not convert the code to python, hence we need to use Matlab. 

### CycleGAN based Method
Find the script for the implementation under: /Users/adon/Desktop/Fall 2025 UCSD/ECE 253/ECE253Project/dehazing/nighttime_dehaze_GAN. This work was also adopted from the paper: > [Enhancing Visibility in Nighttime Haze Images Using Guided APSF and Gradient Adaptive Convolution](https://arxiv.org/abs/2308.01738)\
> ACM International Conference on Multimedia (`ACMMM2023`)\
>[Yeying Jin](https://jinyeying.github.io/), [Beibei Lin](https://bb12346.github.io/), Wending Yan, Yuan Yuan, Wei Ye, and [Robby T. Tan](https://tanrobby.github.io/pub.html)
> 
>[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2308.01738)
>[[Paper]](https://dl.acm.org/doi/10.1145/3581783.3611884).

We thank the authors for making their code open-source. In our group project since the CycleGAN part was not published by the authors we had reimplement the CycleGAN from the official CycleGAN code. You can find our implementation with some modifications such as Wavelet based loss and the changing upsampling methods and other methods that were described in the project report under the dehazing/nighttime_dehaze_GAN/pytorch-CycleGAN-and-pix2pix/models/custom_losses.py. 

To run the script we have provided different weights in our google drive link mentioned below. The pretrained weights are located under the https://drive.google.com/drive/folders/1tXv8FD4gdsDvcAMmJ56W4FUdHq-8MBa7?usp=share_link. The folder has 3 pretrained weights for GTA dataset from authors, Dehazing by authors, modified version by us with custom losses on the real datasets, and another with the GTA dataset.
To evaluate on the real dataset use: https://drive.google.com/drive/folders/1p8w6wY7VkGg9awh0vXup-F-GCYj86vun?usp=sharing. This dataset includes data with our images and also data from opensource resources. To see our images only please go under the collected images folder in the Drive. 

To run the models provided by authors follow the readme inside the GAN based model which was provided by the authors of the work. If you need to run the models trained by us use the following:


python cyclegan_inference.py --input_dir path-to-dataset-from-the-drive --output_dir path-to-output --checkpoint_path path-to-the-weights --size 512 --gpu 0

Inference images were also provided inside the drive under than name "Gan based results" where there are examples of our runs which had a lot of artifacts and better ones under the result 1 and 2 folders. Moreover, you can also find the results for the traditional nighttime dehazing results under the folder traditional_dehazed
## Noiseprint
scripts to run it are in the data folder, this way you can run it as batches. To replicate the paper figures, run the dataset creation python code, then the heatmap script, then computeMAP.py


Check the google drive for the data:  
https://drive.google.com/drive/folders/1512gBQHvucFLopHepM7AWRGZUiZr5mG6?usp=drive_link






