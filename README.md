# product_image_segmentation

| <img src=data/modanet_epoch6/10390187_22dcdd79fac36a4a2a0c8bb3ca3182c1.jpg width=375> | <img src=data/modanet_epoch6/10579132_7dc26dd4ae038798197e215414a63b99.jpg width=375>     | <img src=data/modanet_epoch6/10106352_2d963aa4c36c1ec56d98ab44929efca8.jpg width=375>     |
| ---------- | -----------  | ----------- |
| <img src=data/modanet_epoch6/10480277_455f6d7c7ed86b963f2d46f755c6f535.jpg width=375>     | <img src=data/modanet_epoch6/10477837_365301c24f74f35fbb28fe631245604c.jpg width=375>     | <img src=data/modanet_epoch6/10470870_0e3ffd27c45381ba304f373fc2e8d77f.jpg width=375>     |
| <img src=data/modanet_epoch6/10736337_c24d0e25446071004d503ed92815d763.jpg width=400>     | <img src=data/modanet_epoch6/10842189_582a464aabc178092782415a6d13f08c.jpg width=400>     | <img src=data/modanet_epoch6/10579132_bdfc4e6a76d56c9a35fa41deca412846.jpg width=400>     |



# Algorithm

  In order to do image segmentation and object detection at the same time, and train the data within 2 weeks and get a good enough model, I chose Mask RCNN with Resnet 50 as backbone.\
  I modified [matterport Mask RCNN](https://github.com/matterport/Mask_RCNN) to support training with annotation from [ebay Modanet](https://github.com/eBay/modanet) and images from [paperdoll](https://github.com/kyamagu/paperdoll)

# DataSet
   data sources:
  * [imaterialist-fashion-2019-FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6)
  * [paperdoll](https://github.com/kyamagu/paperdoll)
  * [ebay Modanet](https://github.com/eBay/modanet)
  * DIY dateset
  
  over 500,000 masks and over 100,000 images are used in the training process.
  
# Training Process

  I deployed 2 servers with GPUs in Google Cloud, I mainly used jupyter notebook in this repo to explore and train the data, after each training , I applied the model to the shopee image set to check the predicted results, and improved the datatsets and algorithms accordingly. I trained the models on GPU more than 700 hours GPU hours, with at most 8 K80 GPUs worked together.\
  I trained the models on [imaterialist-fashion-2019-FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6) dataset with pretrained COCO H5 file, due to the mismatch between the annotation and our requirement, the result was not good enough, the records are in MaskR-CNN-FGVC6-resnet50.ipynb and MaskR-CNN-FGVC6-resnet50-aug.ipynb.\
  I made DIY dataset myself, crawled images from google and bing, and I developed a tool to automatically make annotations and masks. But due to the slow process and the time limit, I just add few images into the dataset I used.\
  Finally, I use Moda Net annotation with paperdoll image dataset, with pretrained model from FGVC6.
 

# The Result
  The mAP @ IoU=50 of  epoch 27 model is around 0.4993029372193688.\
  I applied the epoch 6 model of Moda Net to shopee image dataset, results are in the [sample](data/modanet_epoch6) folder.

  
