import csv
import numpy as np
from glob import glob
import os
import datetime
import time

recording_path = "tracking"

def load_data(file_name):
    """ loads a csv """
    with open(file_name) as f:
        reader = csv.reader(f)
        next(reader)
        return [row for row in reader]


def convert_timestamp(data, time_index):
    """ converts a timestamp 01-01-1970 00:00:00 000 into float timestamp 1657180991.372 """
    for date in data:
        time_stamp = date[time_index]
        ms = int(time_stamp.split(' ')[-1]) / 1000
        new_time = time.mktime(datetime.datetime.strptime(time_stamp, "%d-%m-%Y %H:%M:%S %f").timetuple())
        date[time_index] = new_time + ms

    return data

def save_csv(file_name, data):
    """ saves data to csv """
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fileName', 'ObjectName', 'Position', 'Rotation', 'Occlusion', 'Delta_Time'])
        writer.writerows(data)

def multi_conversion():
    # load all scenarios
    scenarios = glob(os.path.join(recording_path, '*'))
    for scenario in scenarios:
        # check if there is a vicon position file for it
        if os.path.exists(f'{scenario}.csv'):
            print('Found', scenario)

            # load vicon data and transform timestamps
            gt_data = load_data(f'{scenario}.csv')
            gt_data = convert_timestamp(gt_data, time_index=1)
        
            # load all cameras that recorded during the scenario
            cameras = glob(os.path.join(scenario, 'camera*'))
            for camera in cameras:
                # source and target files
                file_name = os.path.join(camera, 'data.csv')
                result_file_name = os.path.join(camera, 'annotation.csv')

                # load the image -> timestamp tuple dataset
                try:
                    data = load_data(file_name)
                except:
                    continue
                
                # init counter for optimization to not go through the entire dataset again
                counter = 0
                # each vicon frame incluedes multiple objects remember the first to know when to stop
                first_object = gt_data[0][0]
                new_data = []

                # go throuh all images in the scenario
                for image in data:
                    file_name, timestamp = image
                    file_name = os.path.join(*file_name.split('/')[-3:]).replace('bmp', 'jpg')

                    # if there are no vicon positions for the current timestamp skip this image
                    if float(timestamp) <= gt_data[counter][1]:
                        # print(f'Skipping {file_name} because there is no reference...')
                        continue

                    # continue until the next vicon and image data match
                    while counter < len(gt_data):
                        reference_time = float(gt_data[counter][1])
                        if float(timestamp) >= reference_time:
                            counter+=1
                        else:
                            break
                    
                    # the current counter object is now the stopping object
                    stop_object = gt_data[counter][0]
                    counter-=1
                    # go back to the last position of the stop_object
                    while gt_data[counter][0] != stop_object:
                        counter-=1

                    new_data.append([file_name, gt_data[counter][0], gt_data[counter][2], gt_data[counter][3], gt_data[counter][4], float(timestamp) - reference_time])
                    counter += 1
                    while gt_data[counter][0] != stop_object:
                        new_data.append([file_name, gt_data[counter][0], gt_data[counter][2], gt_data[counter][3], gt_data[counter][4], float(timestamp) - reference_time])
                        counter += 1

                # store everything to csv in the correct folder
                save_csv(result_file_name, new_data)
                print("     Result: ", len(new_data), result_file_name)

def single_conversiton(scenario):
    """ Single dataset convertion """
    # check if there is a vicon position file for it
    if os.path.exists(f'{scenario}.csv'):
        print('Found', scenario)

        # load vicon data and transform timestamps
        gt_data = load_data(f'{scenario}.csv')
        gt_data = convert_timestamp(gt_data, time_index=1)

        # load all cameras that recorded during the scenario
        cameras = glob(os.path.join(scenario, 'camera*'))
        for camera in cameras:
            # source and target files
            file_name = os.path.join(camera, 'data.csv')
            result_file_name = os.path.join(camera, 'annotation.csv')

            # load the image -> timestamp tuple dataset
            data = load_data(file_name)
            
            # init counter for optimization to not go through the entire dataset again
            counter = 0
            # each vicon frame incluedes multiple objects remember the first to know when to stop
            new_data = []

            # go throuh all images in the scenario
            for image in data:
                file_name, timestamp = image
                file_name = os.path.join(*file_name.split('/')[-3:]).replace('bmp', 'jpg')

                # if there are no vicon positions for the current timestamp skip this image
                if float(timestamp) <= float(gt_data[counter][1]):
                    # print(f'Skipping {file_name} because there is no reference...')
                    continue

                # continue until the next vicon and image data match
                while counter < len(gt_data):
                    reference_time = float(gt_data[counter][1])
                    if float(timestamp) >= reference_time:
                        counter+=1
                    else:
                        break

                if counter >= len(gt_data):
                    break
                
                # the current counter object is now the stopping object
                stop_object = gt_data[counter][0]
                counter-=1
                # go back to the last position of the stop_object
                while gt_data[counter][0] != stop_object:
                    counter-=1

                new_data.append([file_name, gt_data[counter][0], gt_data[counter][2], gt_data[counter][3], gt_data[counter][4], float(timestamp) - reference_time])
                counter += 1
                while gt_data[counter][0] != stop_object:
                    new_data.append([file_name, gt_data[counter][0], gt_data[counter][2], gt_data[counter][3], gt_data[counter][4], float(timestamp) - reference_time])
                    counter += 1

            # store everything to csv in the correct folder
            save_csv(result_file_name, new_data)
            print("     Result: ", len(new_data), result_file_name)

if __name__ == "__main__":
    # print(convert_timestamp([['07-07-2022 11:52:14 086']],0))
    # multi_conversion()

    single_conversiton("tracking_dataset_scenario_3_without_dist_LVL_3_3x3_11_33-07_07_2022")