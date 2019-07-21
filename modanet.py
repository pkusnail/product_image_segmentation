"""
    # Train a new model starting from pre-trained COCO weights
    python3 modanet.py train --dataset=/path/to/odanet/ --model=coco

    # Continue training a model that you had trained earlier
    python3 modanet.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Make inference to image set
    python3 modanet.py inference --model=/path/to/weights.h5  --ins=/path/to/input/images --outss=/path/to/save/images

    # Run COCO style evaluatoin on the  model
    python3 modanet.py evaluate --model=/path/to/weights.h5 
"""

import os
import io
import sys
import json
import time
import datetime

import numpy as np
import pandas as pd
import random
import itertools
from tqdm import tqdm

import cv2
import skimage.draw
from PIL import Image
import skimage.io as imio
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

from sklearn.model_selection import StratifiedKFold, KFold
from pycocotools.coco import COCO

import lmdb
import sqlite3
from IPython.display import display

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from pycocotools import mask as maskUtils

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from tensorflow.contrib.tensorboard.plugins import projector

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

################################ GLOBAL VARIABLES ############################
########## DATA ##########

MODANET_ANNO = "./modanet/annotations/modanet2018_instances_train.json"
MODANET_VAL =  "./modanet/annotations/modanet2018_instances_val.json"
PAPERDOLL_META_DB =  "./data/paperdoll/chictopia.sqlite3"
PAPERDOLL_IMG_DB =  "./data/paperdoll/photos.lmdb"


########## MODEL ##########
JOB_NAME = "MODANET"
MODEL_TYPE = 'resnet50'
NUM_CATS = 13
IMAGE_SIZE = 512 #INPUT IMAGE SIZE : IMAGE_SIZE * IMAGE_SIZE

GPU_NUM = 1 #1# 2# 4# #8
IMAGES_PER_GPU_PER_IT = 4 #4 


STEPS_PER_EPOCH_NUM = 500 #40000
VALIDATION_STEPS_NUM = 100


LR = 1e-4
#EPOCHS = [2, 4, 8, 16, 32, 64]
EPOCHS = [2, 4, 8, 9, 10, 11]

N_FOLDS = 5
SELECT_FOLD_IDX = 3

## pretrain data
PRETRAIN_WEIGHT_FILE =  os.path.join(CURR_PATH, "weight_files/logs/modanet20190718T1840/mask_rcnn_modanet_0007.h5")
LOG_DIR =  os.path.join(CURR_PATH, "weight_files/logs")


 
############################################################
#  Configurations
############################################################
class ModaNetConfig(Config):
    def __init__(self, class_num=NUM_CATS, config_name=JOB_NAME):
        self.NUM_CLASSES = class_num + 1 # +1 for the background class
        self.NAME = config_name
        super().__init__()
      
    BACKBONE = MODEL_TYPE
    """     
    GPU_COUNT = GPU_NUM
    IMAGES_PER_GPU = IMAGES_PER_GPU_PER_IT 

    
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE    

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    
    STEPS_PER_EPOCH = STEPS_PER_EPOCH_NUM

    # Skip detections with < 90% confidence
    #DETECTION_MIN_CONFIDENCE = 0.9
    """    

#Photo data in LMDB
class PhotoData(object):
    def __init__(self, path):
        self.env = lmdb.open(
            path, map_size=2**36, readonly=True, lock=False
        )
        
    def __iter__(self):
        with self.env.begin() as t:
            with t.cursor() as c:
                for key, value in c:
                    yield key, value
        
    def __getitem__(self, index):
        key = str(index).encode('ascii')
        with self.env.begin() as t:
            data = t.get(key) # binary image data
        if not data:
            return None
        
        with io.BytesIO(data) as f:
                image = Image.open(f)
                try:
                    image = image.convert("RGB")
                    image.load()
                except:
                    pass
                
                if None == image:
                    try:
                        print(image.format)
                        image = image.convert("L")
                        image.load()
                    except:
                        pass
                    
        return np.array(image)
    
        
    def __len__(self):
        return self.env.stat()['entries']

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1



############################################################
#  Dataset
############################################################

class ModaNetDataset(utils.Dataset):
    def __init__(self, photo_meta_db, coco, label_names, cat_ids):
        super().__init__(self)
        self.coco = coco
        self.label_names = label_names
        self.catIds = cat_ids
        self.photo_meta_db = photo_meta_db
        
    def load_all_dataset(self):
        # Add classes
        for i, name in enumerate(self.label_names):
            self.add_class("fashion", i+1, name)
            
        # Add images and annotations
        for i, row in self.photo_meta_db.iterrows():
            
            annIds = self.coco.getAnnIds(imgIds=row['id'], catIds=self.catIds, iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            catids, polygons = [], []
            for one in anns:
                catids.append(one['category_id'])
                polygons.extend(one['segmentation'])
                
            
            fp = row['path'].split("?")[0]
            if False == fp.lower().endswith("jpg") and False == fp.lower().endswith("jpeg"):
                continue
                
            self.add_image("fashion", 
                           image_id = i,
                           path = "./data/paperdoll/images" + fp ,
                           labels = catids,
                           polygons = polygons,
                           height = row['height'], width=row['width'],
                           annotations = self.coco.loadAnns(self.coco.getAnnIds(
                            imgIds=row['id'], catIds=self.catIds, iscrowd=None)))
        print("load all data done, type :", self.class_info)
    
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle
    
    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m
    
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "fashion.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(ModaNetDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        return super(self.__class__, self).image_reference(image_id)

############################################################
#  Utils
############################################################
def get_fold(splits, train_df, idx):    
    for i, (train_index, valid_index) in enumerate(splits):
        if i == idx:
            return train_df.iloc[train_index], train_df.iloc[valid_index]
        

def split_dateset(photo_meta_db, N_FOLDS, coco, label_names, cat_ids):
    selected = random.randint(0,10000)%N_FOLDS

    kf = KFold(n_splits=N_FOLDS, shuffle=True)
    splits = kf.split(photo_meta_db) 
 
    train_set, valid_set = get_fold(splits, photo_meta_db, selected)


    train_dataset = ModaNetDataset(train_set, coco, label_names, cat_ids)
    train_dataset.load_all_dataset()
    train_dataset.prepare()
    print("train set size: " , len(train_set), type(train_dataset))

    valid_dataset = ModaNetDataset(valid_set, coco, label_names, cat_ids)
    valid_dataset.load_all_dataset()
    valid_dataset.prepare()
    print("valid set size: " , len(valid_set), type(valid_dataset))
    
    return train_dataset, valid_dataset

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois

def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
    return img


############################################################
#  Training
############################################################

def train_model():
    ####### Model Config
    config = ModaNetConfig()
    #config.display()

    ####### prepare Photo Meta Data

    access_str = "file:" + PAPERDOLL_META_DB + "?mode=ro"
    meta_db = sqlite3.connect(access_str, uri=True)
    photo_meta_db = pd.read_sql("""
                SELECT
                    *
                FROM photos
                WHERE photos.post_id IS NOT NULL AND file_file_size IS NOT NULL
            """, con=meta_db)

    ####### prepare Annotation Data
    coco = COCO(MODANET_ANNO)

    cats = coco.loadCats(coco.getCatIds())
    label_names = [cat['name'] for cat in cats]
    cat_ids = coco.getCatIds(catNms = label_names)

    anno_img_id_set = []
    for i in range(len(cat_ids)):
        anno_img_id_set.extend(coco.getImgIds(catIds=[i]))
    print("total annotation number: ", len(anno_img_id_set))

    # drop  images meta entries that without annotation
    anno_img_id_pd = pd.DataFrame({'id':anno_img_id_set})
    anno_img_id_pd['id'] =anno_img_id_pd['id'].apply(int)
    photo_meta_db = pd.merge(photo_meta_db, anno_img_id_pd, how='inner', on=['id'])
    # drop duplicate
    photo_meta_db = photo_meta_db.drop_duplicates()
    print("photo_meta_db size: ", len(photo_meta_db))



    dataset = ModaNetDataset(photo_meta_db, coco, label_names, cat_ids)
    dataset.load_all_dataset()
    dataset.prepare()


    ################################ load pretrain weights
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=LOG_DIR)
    model.load_weights(PRETRAIN_WEIGHT_FILE, by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])

    heades_epochs = 4
    train_set, valid_set = split_dateset(photo_meta_db, N_FOLDS, coco, label_names, cat_ids)
    model.train(train_set, valid_set,
            learning_rate = LR ,
            epochs = heades_epochs,
            layers = 'heads',
            augmentation=None)

    epo = 0
    LRs = [20, 15, 5, 10, 3, 5, 0.7, 0.3, 13, 0.7, 11, 17, 0.7, 0.3,20, 15, 5, 10]
    while epo < len(LRs):
        train_set, valid_set = split_dateset(photo_meta_db, N_FOLDS, coco, label_names, cat_ids)
        model.train(train_set, valid_set,
            learning_rate = LR * LRs[epo],
            epochs = epo + heades_epochs,
            layers = 'all',
            augmentation=None)
        print("One epoch done, NO.: {}  LR: {}".format(epo, LRs[epo]))
        epo = epo + 1

############################################################
#  Inference
############################################################

def model_inference(model_weights_path, 
                    input_folder,
                    output_folder,
                    label_names = ['bag', 'belt', 'boots', 'footwear', 'outer', 'dress', 'sunglasses', 'pants', 'top', 'shorts', 'skirt', 'headwear', 'scarf/tie'], 
                    disp_img=True,
                    log_dir="."):
    infer_config = InferenceConfig()
    #infer_config.display()

    model = modellib.MaskRCNN(mode='inference', 
                          config=infer_config,
                          model_dir=log_dir)
 
    assert model_weights_path != '', "Provide path to load trained weights"
    print("Loading weights from ", model_weights_path)
    model.load_weights(model_weights_path, by_name=True)

    counter = 0 
    for one in os.listdir(input_folder):
        if counter %10 == 0:
            print(counter)
            
        counter += 1
        image_path = str(input_folder + one)
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        result = model.detect([resize_image(image_path)])
        r = result[0]
        
        if r['masks'].size > 0:
            masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
            for m in range(r['masks'].shape[-1]):
                masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                            (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            y_scale = img.shape[0]/IMAGE_SIZE
            x_scale = img.shape[1]/IMAGE_SIZE
            rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
            
            masks, rois = refine_masks(masks, rois)
        else:
            masks, rois = r['masks'], r['rois']
         
        visualize.display_instances(img, rois, masks, r['class_ids'], 
                                    ['bg'] + label_names, r['scores'],
                                    title = one, figsize = (12, 12), save_path = os.path.join(output_folder , one), 
                                    disp_img = disp_img)




    

############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type=None, limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    if None != eval_type:
        cocoEval = COCOeval(coco, coco_results, eval_type)
        cocoEval.params.imgIds = coco_image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        print("Prediction time: {}. Average {}/image".format(
            t_prediction, t_prediction / len(image_ids)))
        print("Total time: ", time.time() - t_start)

    else:
        bboxEval = COCOeval(coco, coco_results, "bbox")
        bboxEval.params.imgIds = coco_image_ids
        bboxEval.evaluate()
        bboxEval.accumulate()
        bboxEval.summarize()

        print(" bbox EvalPrediction time: {}. Average {}/image".format(
            t_prediction, t_prediction / len(image_ids)))
        print("Total time: ", time.time() - t_start)

        segmEval = COCOeval(coco, coco_results, "segm")
        segmEval.params.imgIds = coco_image_ids
        segmEval.evaluate()
        segmEval.accumulate()
        segmEval.summarize()

        print(" segm EvalPrediction time: {}. Average {}/image".format(
            t_prediction, t_prediction / len(image_ids)))
        print("Total time: ", time.time() - t_start)
        

def evaluate_model(model_weights_path, eval_type=None, limit=0, image_ids=None, log_dir="."):
    ####### prepare Photo Meta Data

    access_str = "file:" + PAPERDOLL_META_DB + "?mode=ro"
    meta_db = sqlite3.connect(access_str, uri=True)
    photo_meta_db = pd.read_sql("""
                SELECT
                    *
                FROM photos
                WHERE photos.post_id IS NOT NULL AND file_file_size IS NOT NULL
            """, con=meta_db)

    ####### prepare Annotation Data
    coco = COCO(MODANET_ANNO)

    cats = coco.loadCats(coco.getCatIds())
    label_names = [cat['name'] for cat in cats]
    cat_ids = coco.getCatIds(catNms = label_names)

    anno_img_id_set = []
    for i in range(len(cat_ids)):
        anno_img_id_set.extend(coco.getImgIds(catIds=[i]))
    print("total annotation number: ", len(anno_img_id_set))

    # drop  images meta entries that without annotation
    anno_img_id_pd = pd.DataFrame({'id':anno_img_id_set})
    anno_img_id_pd['id'] =anno_img_id_pd['id'].apply(int)
    photo_meta_db = pd.merge(photo_meta_db, anno_img_id_pd, how='inner', on=['id'])
    # drop duplicate
    photo_meta_db = photo_meta_db.drop_duplicates()
    print("photo_meta_db size: ", len(photo_meta_db))

    dataset = ModaNetDataset(photo_meta_db, coco, label_names, cat_ids)
    dataset.load_all_dataset()
    dataset.prepare()

    ##### prepare model
    infer_config = InferenceConfig()
    model = modellib.MaskRCNN(mode='inference', 
                          config=infer_config,
                          model_dir=log_dir)
    
 
    assert model_weights_path != '', "Provide path to load trained weights"
    print("Loading weights from ", model_weights_path)
    model.load_weights(model_weights_path, by_name=True)

    evaluate_coco(model, dataset, coco, eval_type, limit, image_ids)
    


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for ModaNet Dataset')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'inference' or 'evaluate'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file ")
    parser.add_argument('--logs', required=False,
                        default="." ,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default="500" ,
                        metavar="500",
                        help='number of images to evaluate the model')
    parser.add_argument('--ins', required=False,
                        default="." ,
                        metavar="/path/to/shopee/images",
                        help='the path to shopee images')
    parser.add_argument('--outs', required=False,
                        default="./out" ,
                        metavar="/path/to/store/processed/shopee/images",
                        help='the path to save processed shopee images')
    parser.add_argument('--eval_type', required=False,
                        default="None" ,
                        metavar="bbox or segm",
                        help='evaluation type')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Limit: ", args.limit)
    print("In: ", args.ins)
    print("Out: ", args.outs)


    if  "train" == args.command:
        train_model()
    elif "inference" == args.command:
        label_names = ['bag', 'belt', 'boots', 'footwear', 'outer', 'dress', 'sunglasses', 'pants', 'top', 'shorts', 'skirt', 'headwear', 'scarf/tie'], 
                   
        model_inference(args.model, 
                    args.ins,
                    args.outs,
                    disp_img = True,
                    label_names = label_names,
                    log_dir=".")

    elif "evaluate" == args.command:
        eval_type = None
        if "None" != args.eval_type:
            eval_type = args.eval_type
        
        evaluate_model(args.model, eval_type=eval_type, limit=args.limit, image_ids=None, log_dir=".")
        
    else:
        print("'{}' is not recognized. Use 'train' , 'inference' or 'evaluate'".format(args.command))


