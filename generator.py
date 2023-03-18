import copy
import os
import csv
import ast
import time
import numpy as np
import glob
import cv2

from icecream import ic
from bop_toolkit import dataset_params, renderer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

cams_ids = [3, 4, 5, 6, 7]
SAVE_ANNOTATION = True
SAVE_OVERLAY = False
cam_locations_csv_file = "./camera_locations.csv"
calib_params_path = "./camera_calib.csv"
dataset_path = "/home/athos/Schreibtisch/tracking_dataset_copy/"


def get_homogenous_form(rot, trans):
    mat = np.column_stack((rot, trans))
    mat_homog = np.row_stack((mat, [0.0, 0.0, 0.0, 1.0]))
    return mat_homog


def invert_homog_transfrom(homog_trans):
    trans = homog_trans[0:3, 3]
    rot = homog_trans[0:3, 0:3]
    rot_inv = np.linalg.inv(rot)
    homog_inv = get_homogenous_form(rot_inv, -1 * (rot_inv.dot(trans)))
    return homog_inv

p = {
    # See dataset_params.py for options.
    'dataset': 'tracking',

    # Dataset split. Options: 'train', 'val', 'test'.
    'dataset_split': 'test',

    # Dataset split type. None = default. See dataset_params.py for options.
    'dataset_split_type': None,

    # Type of the renderer.
    'renderer_type': 'vispy',  # Options: 'vispy', 'cpp', 'python'.

    # Folder containing the BOP datasets.
    # """tracking dataset"""
    'datasets_path': dataset_path,
}

ic.disable()

# Load dataset parameters.
dp_split = dataset_params.get_split_params(
    p['datasets_path'], p['dataset'], p['dataset_split'], p['dataset_split_type'])

im_width, im_height = dp_split['im_size']
ren_width, ren_height = 3 * im_width, 3 * im_height
ren_cx_offset, ren_cy_offset = im_width, im_height
ren = renderer.create_renderer(
    ren_width, ren_height, p['renderer_type'], mode='depth')

object_color_mapping = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 0),
    3: (0, 255, 255),
    4: (255, 0, 0),
    5: (255, 0, 255),
    6: (255, 255, 0)
}

camera2vicon = {}
with open(cam_locations_csv_file) as f:
    reader = csv.reader(f)
    reader_list = list(reader)
    for row in reader_list:
        cameraId = row[0]
        translation = np.array(ast.literal_eval(row[1]))
        rotation = np.array(ast.literal_eval(row[2]))  # rotation matrix
        transfrom = get_homogenous_form(rotation, translation)
        camera2vicon[cameraId] = transfrom

dp_model = dataset_params.get_model_params(
    p['datasets_path'], p['dataset'], None)
for obj_id in dp_model['obj_ids']:
    model_fpath = dp_model['model_tpath'].format(obj_id=obj_id)
    ren.add_object(obj_id, model_fpath)


def generate_gt_transformation(cam_id, obj2vicon_trans, obj2vicon_rot):
    cam2vicon_transform = camera2vicon[str(cam_id)]
    obj2vicon_rot = R.from_euler('XYZ', obj2vicon_rot, degrees=False)
    obj2vicon_rot_mat = obj2vicon_rot.as_matrix()

    obj2vicon_transform = get_homogenous_form(
        obj2vicon_rot_mat, obj2vicon_trans)
    ic(obj2vicon_transform)

    obj2cam_transform = invert_homog_transfrom(
        cam2vicon_transform).dot(obj2vicon_transform)
    ic(obj2cam_transform)
    obj2cam_trans = obj2cam_transform[0:3, 3]
    obj2cam_rot = obj2cam_transform[0:3, 0:3]

    return obj2cam_trans, obj2cam_rot


def read_calib_params(calib_path):
    '''
        Loads csv file with calib params and saves it to a dict
    '''
    out_dict = {}
    content = {}
    with open(calib_path) as f:
        reader = csv.reader(f)
        reader_list = list(reader)
        keys = reader_list[0]
        for row in reader_list[1:]:
            content['cam_K'] = np.asarray(
                ast.literal_eval(row[keys.index('cam_K')]))
            content['depth_scale'] = int(row[keys.index('depth_scale')])
            content_copy = copy.deepcopy(content)
            out_dict[row[keys.index('cam_id')]] = content_copy
    return out_dict


def generate_depth_image(gt, cam_id):
    scene_camera = read_calib_params(calib_params_path)
    K = scene_camera[str(cam_id)]['cam_K']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # Render depth image of the object model in the ground-truth pose.
    depth_gt_large = ren.render_object(
        gt['obj_id'], gt['cam_R_m2c'], gt['cam_t_m2c'],
        fx, fy, cx + ren_cx_offset, cy + ren_cy_offset)['depth']
    depth_gt = depth_gt_large[
        ren_cy_offset:(ren_cy_offset + im_height),
        ren_cx_offset:(ren_cx_offset + im_width)]
    if np.sum(depth_gt) < 100 or np.sum(depth_gt) < 0.9 * np.sum(depth_gt_large):
        return None
    return depth_gt


def calc_2d_bbox(img):
    x_min = -1
    x_max = -1
    y_min = -1
    y_max = -1
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] > 0:
                if x > x_max:
                    x_max = x
                if x < x_min or x_min == -1:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min or y_min == -1:
                    y_min = y

    return [x_min, y_min, x_max-x_min, y_max-y_min]


def safe_create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    path = os.path.join(p["datasets_path"], "dataset", '*')
    recordings = glob.glob(path)

    for recording_path in recordings:
        if ".zip" in recording_path or ".ipynb" in recording_path or ".csv" in recording_path:
            continue

        print(recording_path)
        for cam_id in cams_ids:

            cam_path = os.path.join(recording_path, 'camera_' + str(cam_id))
            print("     ", cam_path)
            safe_create_folder(os.path.join(cam_path, 'annoated_imgs'))
            safe_create_folder(os.path.join(cam_path, 'overlay_imgs'))
            data = []
            boundingbox_data = []
            depth_image_data = []
            previouse_img_name = ""
            with open(os.path.join(cam_path, 'new_data.csv')) as f:
                reader = csv.reader(f)
                reader_list = list(reader)
                keys = reader_list[0]
                for row in tqdm(reader_list[1:]):
                    start_time = time.time()
                    obj2vicon_trans = np.asarray(
                        ast.literal_eval(row[keys.index('Position')]))
                    obj2vicon_rot = np.asarray(
                        ast.literal_eval(row[keys.index('Rotation')]))
                    camToObjTrans, cam_R_m2c = generate_gt_transformation(
                        cam_id, obj2vicon_trans, obj2vicon_rot)

                    object_name = row[keys.index('ObjectName')]
                    image_name = os.path.split(
                            row[keys.index('fileName')])[-1]

                    """tracking dataset"""
                    # pallet        -> 0
                    # forklift      -> 1    
                    # klt           -> 2
                    # barrel        -> 3
                    # box_big       -> 4    Box_Track_2
                    # box_small     -> 5    Box_Track_1
                    # gitter_box    -> 6

                    object_id = None
                    if "PALLET" in object_name:
                        object_id = 0
                    elif "FORKLIFT" in object_name:
                        object_id = 1
                    elif "KLT" in object_name:
                        object_id = 2
                    elif "Barrel" in object_name:
                        object_id = 3
                    elif "BOX_TRACK_2" == object_name:
                        object_id = 4
                    elif "BOX_TRACK_1" == object_name:
                        object_id = 5
                    elif "Gitter" in object_name:
                        object_id = 6
                
                    if object_id is None or object_id > 6:
                        continue

                    entry = {'cam_t_m2c': camToObjTrans,
                             'cam_R_m2c': cam_R_m2c, 'obj_id': int(object_id)}
                    depth_image = generate_depth_image(entry, cam_id)
                    occlusion = int(row[keys.index('Occlusion')])
                    if depth_image is not None and not occlusion:
                        depth_image = cv2.flip(depth_image, 0)
                        depth_image = cv2.flip(depth_image, 1)
                        bb = calc_2d_bbox(depth_image)

                        duration = time.time() - start_time
                        data.append([*row, bb, 1, duration])

                        if previouse_img_name == image_name:
                            boundingbox_data.append([*bb, object_id])
                            depth_image_data.append(np.stack((depth_image,)*3, axis=-1))
                        elif len(boundingbox_data) == 0:
                            boundingbox_data = [[*bb, object_id]]
                            depth_image_data = [np.stack((depth_image,)*3, axis=-1)]
                            previouse_img_name = image_name

                    else:
                        duration = time.time() - start_time
                        data.append([*row, [-1, -1, -1, -1], 0, duration])


                    if previouse_img_name != image_name and len(boundingbox_data) > 0:

                        img = cv2.imread(os.path.join(
                            cam_path, 'images', previouse_img_name))

                        if img is None:
                            continue

                        if SAVE_OVERLAY:
                            final_depth_image = depth_image_data[0]
                            for depth_img in depth_image_data[1:]:
                                final_depth_image = np.add(final_depth_image, depth_img)

                            final_depth_image[final_depth_image >= 100] = 255
                            comb_img = cv2.add(img, final_depth_image.astype(np.uint8))
                            cv2.imwrite(os.path.join(
                                cam_path, 'overlay_imgs', f"overlay_mask_{previouse_img_name}"), final_depth_image)
                            cv2.imwrite(os.path.join(
                                cam_path, 'overlay_imgs', f"overlay_{previouse_img_name}"), comb_img)

                        if SAVE_ANNOTATION:
                            for bbox in boundingbox_data:
                                start_point = (bbox[1], bbox[0])
                                end_point = (bbox[1]+bbox[3], bbox[0]+bbox[2])
                                img = cv2.rectangle(
                                    img, start_point, end_point, object_color_mapping[bbox[4]], 5)

                            cv2.imwrite(os.path.join(
                                cam_path, 'annoated_imgs', f"annotation_{previouse_img_name}"), img)

                        boundingbox_data = []
                        depth_image_data = []
                        previouse_img_name = image_name
                    elif previouse_img_name != image_name:
                        previouse_img_name = image_name

                    

            with open(os.path.join(cam_path, 'annotated_data.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["fileName", "ObjectName", "Position",
                                 "Rotation", "Occlusion", "Delta_Time", "BoundingBox", "Visible", "AnnotationTime"])
                writer.writerows(data)


if __name__ == "__main__":
    main()
