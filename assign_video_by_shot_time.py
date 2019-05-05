from source.tools import get_format_file
import os
from os.path import join
import yaml
import re
from datetime import datetime

project_dir = os.getcwd()
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

origin_data_path = join(project_dir, cfg['base_conf']['origin_data_path'])

folder_names = os.listdir(origin_data_path)

folder_names = list(filter(lambda x: re.match(r'^\d{2}-\d{2}', x), folder_names))
time_folders = []
for folder_name in folder_names:
    times = folder_name.split('_')
    time_folders.append([datetime.strptime(times[0], '%m-%d-%H-%M-%S'),
                         datetime.strptime(times[1], '%m-%d-%H-%M-%S'), folder_name])
    time_folders.sort(key=lambda x: x[0])

videos = os.listdir(join(project_dir, 'video'))
videos.sort(key=lambda x: datetime.strptime(x.split('.')[0].split('_')[0], '%m-%d-%H-%M-%S'))

index = 0
for video in videos:
    video_time = datetime.strptime(video.split('.')[0].split('_')[0], '%m-%d-%H-%M-%S')
    video_time.year
    if index >= len(time_folders):
        break
    if video_time >= time_folders[index][0]:
        while True:
            if index >= len(time_folders):
                break
            if video_time <= time_folders[index][1] and  video_time >= time_folders[index][0]:
                os.rename(join(project_dir, 'video', video), join(origin_data_path, time_folders[index][2], video))
                break
            elif video_time < time_folders[index][0]:
                break
            else:
                index += 1





