import csv
from glob import glob
import os
import shutil
import ast
import numpy as np
from tqdm import tqdm


def get_id(object_name):
    if "PALLET" in object_name:
        return 0
    elif "FORKLIFT" in object_name:
        return 1
    elif "KLT" in object_name:
        return 2
    elif "Barrel" in object_name:
        return 3
    elif "BOX_TRACK_2" == object_name:
        return 4
    elif "BOX_TRACK_1" == object_name:
        return 5
    elif "Gitter" in object_name:
        return 6

def safe_create_folder(folderName):
    if not os.path.exists(folderName):
        os.mkdir(folderName)
    return folderName

def create_folder_name(folderName):
    if "9_43" in folderName:
        return "LOG22-00-9_43"
    elif "9_58" in folderName:
        return "LOG22-00-9_58"
    elif "10_00" in folderName:
        return "LOG22-00-10_00"
    elif "11_52" in folderName:
        return "LOG22-00-11_52"
    elif "15_39" in folderName:
        return "LOG22-00-15_39"
    elif "15_52" in folderName:
        return "LOG22-00-15_52"
    else:
        return folderName


def main():
    datasets = glob('dataset/*')

    root = safe_create_folder(os.path.join("..", "mot"))

    for dataset in datasets:
        print(f"Processing dataset: {dataset}...")
        folderName = create_folder_name(os.path.split(dataset)[-1])
        data_path = safe_create_folder(os.path.join(root, folderName))
        image_path = safe_create_folder(os.path.join(data_path, "img1"))
        det_path = safe_create_folder(os.path.join(data_path, "det"))
        gt_path = safe_create_folder(os.path.join(data_path, "gt"))
        # annotation_path = safe_create_folder(os.path.join(data_path, "annotations"))


        images = glob(os.path.join(dataset, "camera_6", "images", '*.jpg'))
        images.sort(key=lambda img: int(os.path.split(img)[-1].split('.')[0]))
        first_img_id = int(os.path.split(images[0])[-1].split('.')[0])
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
                    bb[1],
                    bb[0],
                    bb[3],
                    bb[2],
                    row[1],
                    vis
                ])
            data = np.array(data)


        first_id = int(os.path.split(images[0])[-1].split(".")[0])
        print(first_id)
        with open(os.path.join(gt_path, "gt.txt"), "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            for obj_idf in object_ids:
                for row in data[data[:,-2] == obj_idf]:
                    id_ = int(row[0])
                    if id_ >= first_id and ast.literal_eval(row[-1]) > 0:
                        object_id = object_ids[row[-2]]
                    
                        # 1569,1,0,175,234,113,1,1,1
                        writer.writerow([
                            id_ - first_img_id,
                            object_id,
                            *row[2:6],
                            1,
                            row[1],
                            row[-1]
                        ])

        with open(os.path.join(det_path, "det.txt"), 'w', newline="") as f:
            writer = csv.writer(f, delimiter=',')
            for image_name in tqdm(images):
                name = os.path.split(image_name)[-1]
                id_ = int(name.split(".")[0])

                # [318   0 352 165 117 153]
                data_ = data[data[:,0].astype(int) == id_]

                # 481,-1,1005,397,290,104,1,-1,-1,-1
                for date in data_:
                    if ast.literal_eval(date[-1]) > 0:
                        writer.writerow([id_ - first_img_id, -1, *date[2:6], 1, -1, date[1], 1])

                target = os.path.join(image_path, f"{id_ - first_img_id}.jpg")
                if not os.path.exists(target):
                    shutil.copy(image_name, target)

    print(object_ids)


if __name__ == "__main__":
    main()