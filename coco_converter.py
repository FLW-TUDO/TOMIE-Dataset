import csv
from glob import glob
import os
import shutil
import ast
import numpy as np
from tqdm import tqdm
import json
import datetime

CATEGORIES = ["PALLET", "FORKLIFT", "KLT", "Barrel", "BOX_TRACK_2", "BOX_TRACK_1", "Gitter", "NONE"]

def get_id(object_name):
    index = 0
    for index, cat in enumerate(CATEGORIES):
        if cat in object_name:
            return index
    return -1

def safe_create_folder(folderName):
    if not os.path.exists(folderName):
        os.mkdir(folderName)
    return folderName

def create_folder_name(folderName):
    if "9_43" in folderName:
        return "MOT20-02-9_43"
    elif "9_58" in folderName:
        return "MOT20-02-9_58"
    elif "10_00" in folderName:
        return "MOT20-02-10_00"
    elif "11_52" in folderName:
        return "MOT20-02-11_52"
    elif "15_39" in folderName:
        return "MOT20-02-15_39"
    elif "15_52" in folderName:
        return "MOT20-02-15_52"
    elif "11_33" in folderName:
        return "MOT20-02-11_33"
    else:
        return folderName

def generateInfo(): 
    return {
        "year" : 2023,
        "version" : "0.1",
        "description" : "Trial dataset for logistical pallet dataset",
        "contributer" : "",
        "url" : "",
        "date_created" : str(datetime.datetime.now())
    }

def generateCategories():
    return [
        {
            "id": index,
            "name": cat,
            "supercategory": "Logistics",
        } for index, cat in enumerate(CATEGORIES)
    ]

def generateImageInfo(image_name : str):
    return {
        "id" : int(os.path.split(image_name)[-1].split('.')[0]),
        "width" : 1296,
        "height" : 1024,
        "file_name" : f"{int(os.path.split(image_name)[-1].split('.')[0])}.jpg",
        "license" : 0,
        "flickr_url" : "",
        "coco_url" : "",
        "date_captured" : str(datetime.datetime.now())
    }

def generateImageInfoId(image_id : int):
    return {
        "id" : image_id,
        "width" : 1296,
        "height" : 1024,
        "file_name" : f"{image_id}.jpg",
        "license" : 0,
        "flickr_url" : "",
        "coco_url" : "",
        "date_captured" : str(datetime.datetime.now())
    }

def create_indiv_datasets():
    """ Create individual datasets"""
    datasets = glob('dataset/*')

    root = safe_create_folder(os.path.join("..", "coco"))

    for dataset in datasets:
        print(f"Processing dataset: {dataset}...")
        folderName = create_folder_name(os.path.split(dataset)[-1])
        data_path = safe_create_folder(os.path.join(root, folderName))
        train_path = safe_create_folder(os.path.join(data_path, "train"))
        val_path = safe_create_folder(os.path.join(data_path, "val"))
        annotation_path = safe_create_folder(os.path.join(data_path, "annotations"))


        images = glob(os.path.join(dataset, "camera_6", "images", '*.jpg'))
        images.sort(key=lambda img: int(os.path.split(img)[-1].split('.')[0]))
        object_ids = {}
        with open(os.path.join(dataset, "camera_6", "new_data.csv")) as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            data = []
            # fileName ObjectName Position Rotation Occlusion Delta_Time BoundingBox Visible
            for row in reader:
                vis = ast.literal_eval(row[-1])
                bb = ast.literal_eval(row[-2])
                if row[1] not in object_ids:
                    object_ids[row[1]] = len(object_ids)
                # [image_id, object_class, bb_y, bb_x, bb_h, bb_w, object_name, vis]
                data.append([
                    int(os.path.split(row[0])[-1].split('.')[0]),
                    get_id(row[1]),
                    int(bb[1]),
                    int(bb[0]),
                    int(bb[3]),
                    int(bb[2]),
                    row[1],
                    vis
                ])
            data = np.array(data)


        # instances_{train,val}2017.json
        coco_train_dataset = {
            "info" : generateInfo(),
            "images" : [],
            "annotations" : [],
            "license" : [],
            "categories" : generateCategories()
        }

        coco_val_dataset = {
            "info" : generateInfo(),
            "images" : [],
            "annotations" : [],
            "license" : [],
            "categories" : generateCategories()
        }

        annotation_index = 0
        for image_name in tqdm(images):
            name = os.path.split(image_name)[-1]
            id_ = int(name.split(".")[0])

            rnd = np.random.random()
            image_path = train_path if rnd < 0.9 else val_path
            dataset = coco_train_dataset if rnd < 0.9 else coco_val_dataset

            dataset["images"].append(generateImageInfo(image_name))

            # [318   0 352 165 117 153]
            data_ = data[data[:,0].astype(int) == id_]

            # 481,-1,1005,397,290,104,1,-1,-1,-1
            for date in data_:
                if ast.literal_eval(date[-1]) > 0:
                    dataset["annotations"].append(
                        {
                            "id" : annotation_index,
                            "image_id" : id_,
                            "category_id" : int(date[1]),
                            "semgentation" : [],
                            "area" : 1,
                            "bbox" : list([int(x) for x in date[2:6]]),
                            "iscrowd" : 0
                        }
                    )
                    annotation_index += 1

            target = os.path.join(image_path, f"{id_}.jpg")
            if not os.path.exists(target):
                shutil.copy(image_name, target)

        with open(os.path.join(annotation_path, "instances_train.json"), "w") as f:
            json.dump(coco_train_dataset, f)
        with open(os.path.join(annotation_path, "instances_val.json"), "w") as f:
            json.dump(coco_val_dataset, f)
    print(object_ids)


def create_single_dataset():
    """Create one combined dataset"""
    datasets = glob('dataset/*')

    root = safe_create_folder(os.path.join("..", "coco"))
    folderName = "log_coco"
    data_path = safe_create_folder(os.path.join(root, folderName))
    train_path = safe_create_folder(os.path.join(data_path, "train"))
    val_path = safe_create_folder(os.path.join(data_path, "val"))
    annotation_path = safe_create_folder(os.path.join(data_path, "annotations"))


    # instances_{train,val}2017.json
    coco_train_dataset = {
        "info" : generateInfo(),
        "images" : [],
        "annotations" : [],
        "license" : [],
        "categories" : generateCategories()
    }

    coco_val_dataset = {
        "info" : generateInfo(),
        "images" : [],
        "annotations" : [],
        "license" : [],
        "categories" : generateCategories()
    }


    image_id = 0
    annotation_index = 0
    for dataset_path in datasets:
        print(f"Processing dataset: {dataset_path}...")
        images = glob(os.path.join(dataset_path, "camera_6", "images", '*.jpg'))
        images.sort(key=lambda img: int(os.path.split(img)[-1].split('.')[0]))
        with open(os.path.join(dataset_path, "camera_6", "new_data.csv")) as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            data = []
            # fileName ObjectName Position Rotation Occlusion Delta_Time BoundingBox Visible
            for row in reader:
                vis = ast.literal_eval(row[-1])
                bb = ast.literal_eval(row[-2])
                # [image_id, object_class, bb_y, bb_x, bb_h, bb_w, object_name, vis]
                data.append([
                    int(os.path.split(row[0])[-1].split('.')[0]),
                    get_id(row[1]),
                    int(bb[1]),
                    int(bb[0]),
                    int(bb[3]),
                    int(bb[2]),
                    row[1],
                    vis
                ])
            data = np.array(data)

        for image_name in tqdm(images):
            name = os.path.split(image_name)[-1]
            id_ = int(name.split(".")[0])

            rnd = np.random.random()
            image_path = train_path if rnd < 0.9 else val_path
            dataset = coco_train_dataset if rnd < 0.9 else coco_val_dataset

            dataset["images"].append(generateImageInfoId(image_id))

            # [318   0 352 165 117 153]
            data_ = data[data[:,0].astype(int) == id_]

            # 481,-1,1005,397,290,104,1,-1,-1,-1
            for date in data_:
                if ast.literal_eval(date[-1]) > 0:
                    dataset["annotations"].append(
                        {
                            "id" : annotation_index,
                            "image_id" : image_id,
                            "category_id" : int(date[1]),
                            "semgentation" : [],
                            "area" : 1,
                            "bbox" : list([int(x) for x in date[2:6]]),
                            "iscrowd" : 0
                        }
                    )
                    annotation_index += 1

            target = os.path.join(image_path, f"{image_id}.jpg")
            if not os.path.exists(target):
                shutil.copy(image_name, target)
            image_id += 1

    with open(os.path.join(annotation_path, "instances_train.json"), "w") as f:
        json.dump(coco_train_dataset, f)
    with open(os.path.join(annotation_path, "instances_val.json"), "w") as f:
        json.dump(coco_val_dataset, f)

if __name__ == "__main__":
    create_single_dataset()
