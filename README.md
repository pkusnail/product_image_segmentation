# product_image_segmentation

| <img src=data/modanet_epoch6/10390187_22dcdd79fac36a4a2a0c8bb3ca3182c1.jpg width=375> | <img src=data/modanet_epoch6/10579132_7dc26dd4ae038798197e215414a63b99.jpg width=375>     | <img src=data/modanet_epoch6/10106352_2d963aa4c36c1ec56d98ab44929efca8.jpg width=375>     |
| ---------- | -----------  | ----------- |
| <img src=data/modanet_epoch6/10480277_455f6d7c7ed86b963f2d46f755c6f535.jpg width=375>     | <img src=data/modanet_epoch6/10477837_365301c24f74f35fbb28fe631245604c.jpg width=375>     | <img src=data/modanet_epoch6/10470870_0e3ffd27c45381ba304f373fc2e8d77f.jpg width=375>     |
| <img src=data/modanet_epoch6/10736337_c24d0e25446071004d503ed92815d763.jpg width=400>     | <img src=data/modanet_epoch6/10842189_582a464aabc178092782415a6d13f08c.jpg width=400>     | <img src=data/modanet_epoch6/10579132_bdfc4e6a76d56c9a35fa41deca412846.jpg width=400>     |


# Task Understanding
  According to the task description, candidates are required to deliver an end to end working image segmentation system, given an image that contains a single item or multiple items, the system is expected to return the bounding box and ideally item tag.
  
  Although the system is named "image segmentation system", it is basically an object detection task according to the description.
  But taking the recent development of deep learning technologies into consideration, I chose to use the Mask RCNN model to this task.
  It is able to return the bounding box and ideally item tag, and further more ,  return the pixel level mask, which is part of image segmentation task.
  
  
# Model Selection
  There are 2-stage models such as R-CNN、SPP-Net、Fast R-CNN、Faster R-CNN and Mask R-CNN;  and 1-stage models such as OverFeat、YOLOv1、YOLOv2、YOLOv3、SSD和RetinaNet, in general, 2-stage models achieve better precision than 1-stage models, and Mask R-CNN can do image classification, image detection and image segmentation at the same time, so I chose Mask R-CNN to do this task.

  In order to do image segmentation and object detection at the same time, and train the data within 2 weeks and get a good enough model, I chose Mask RCNN with Resnet 50 as backbone.
  
  
# DataSet
  There are several open source datasets available:
  * [imaterialist-fashion-2019-FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6)\
  FGVC6 contains 46 apparel objects (27 main apparel items and 19 apparel parts), and 92 related fine-grained attributes. Secondly, a total of 50K clothing images (10K with both segmentation and fine-grained attributes, 40K with apparel instance segmentation) in daily-life, celebrity events, and online shopping are labeled by both domain experts and crowd workers for fine-grained segmentation.\
  The lables are ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']. The lables of FGVC6 is not quit good for this task, as there are labels such as 'sleeve', 'pocket','collar', which are not required by this task, in order to use this dataset properly, masks with these labels should be deleted.
  
  FGVC6 is part of DeepFashion2.
  * [DeepFashion2](https://github.com/switchablenorms/DeepFashion2)\
   DeepFashion2 is a very good dataset, but I could not get the data until the second last day before the deadline, I was not able to take advantage of it.
   
  * [ebay Modanet](https://github.com/eBay/modanet)\
  Modanet provides annotations for  55,176 street images, fully annotated with polygons on top of the 1 million weakly annotated street images in [paperdoll](https://github.com/kyamagu/paperdoll). 13 meta categories in Modanet dataset are: bag, belt, boots, footwear, outer(coat, jacket, suit, blazers), dress (dress, t-shirt dress), sunglasses, pants(pants, jeans, leggings),top( top, blouse, t-shirt, shirt), shorts, skirt,headwear, scarf&tie(scarf, tie).\
  I mainly used Modanet to train my models.
  
  * DIY dataset
  
  over 500,000 masks and over 100,000 images are used in the training process.
  
# Training Process

  I deployed 2 servers with GPUs in Google Cloud, I mainly used jupyter notebook in this repo to explore and train the data, after each training , I applied the model to the shopee image set to check the predicted results, and improved the datatsets and algorithms accordingly. I trained the models on GPU more than 700 GPU hours, with at most 8 K80 GPUs working together.\
  I trained the models on [imaterialist-fashion-2019-FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6) dataset with pretrained COCO H5 file, due to the mismatch between the annotation and our requirement, the result was not good enough, the records are in MaskR-CNN-FGVC6-resnet50.ipynb and MaskR-CNN-FGVC6-resnet50-aug.ipynb.\
  I made DIY dataset myself, crawled images from google and bing, and I developed a tool to automatically make annotations and masks. But due to the slow process and the time limit, I just add few images into the dataset I used.\
  Finally, I use Moda Net annotation with paperdoll image dataset, with pretrained model from FGVC6.
 

# The Result
  The mAP @ IoU=50 of  epoch 27 model is around 0.4993029372193688.\
  I applied the epoch 6 model to shopee image dataset, results are in the [sample](data/modanet_epoch6) folder. The pictures in this readme are also outputs of the model.

  
