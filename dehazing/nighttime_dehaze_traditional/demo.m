clear all
close all

img=imread('haze_ours/haze_79.jpg');
img = im2double(img);
dh_img=nt_dehaze(img);
imshow([img dh_img]);
imwrite(dh_img, 'haze_79.jpg')