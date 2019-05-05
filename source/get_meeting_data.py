import tools
import os
import yaml
from os.path import join
from shutil import copyfile, rmtree
from datetime import datetime, timedelta

project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

middle_data_path = join(project_dir, cfg['base_conf']['middle_data_path'])
time_data_path = join(middle_data_path, 'time')

start_time = cfg['get_meeting_data']['start_time']
start_time = datetime.strptime(start_time, '%m-%d-%H-%M')
end_time = cfg['get_meeting_data']['end_time']
end_time = datetime.strptime(end_time, '%m-%d-%H-%M')
duration = cfg['get_meeting_data']['duration_time']
duration = timedelta(minutes=duration)
section_time = timedelta(minutes=180)

time_folder_name = os.listdir(time_data_path)

time_folders = []
for folder_name in time_folder_name:
    times = folder_name.split('_')
    time_folders.append([datetime.strptime(times[0], '%m-%d-%H-%M-%S'),
                         datetime.strptime(times[1], '%m-%d-%H-%M-%S'), folder_name])
    time_folders.sort(key=lambda x: x[0])

thres = cfg['pre_process']['wifi_threshold']
meeting_name = tools.get_meeting_people_name(time_data_path, thres)

while True:
    if start_time > time_folders[-1][1]:
        break
    meeting_end_time = start_time + duration
    pic_paths = []
    people_name = set([])
    for times in time_folders:
        pics = []
        if times[0] < meeting_end_time:
            if times[0] >= start_time + timedelta(minutes=20):
                pics = tools.get_format_file(join(time_data_path, times[2]), 2, r'.+\.jpeg')
                if times[2] in meeting_name:
                    people_name |= set(meeting_name[times[2]])
                else:
                    print(times[2])
        else:
            break
        pic_paths.extend(pics)
    folder_name = '%s_%s' % (start_time.strftime('%m-%d-%H-%M-%S'), meeting_end_time.strftime('%m-%d-%H-%M-%S'))
    if os.path.exists(join(middle_data_path, folder_name)):
        rmtree(join(middle_data_path, folder_name))
    os.mkdir(join(middle_data_path, folder_name))
    os.mkdir(join(middle_data_path, folder_name, cfg['base_conf']['mtcnn_origin_data_path']))
    index = 0
    for pic in pic_paths:
        iou_num = pic.split('.')[-2]
        iou_num = iou_num.split('-')[-1]
        meeting_n = tools.get_parent_folder_name(pic, 3)
        copyfile(pic, join(middle_data_path, folder_name, cfg['base_conf']['mtcnn_origin_data_path'],
                           '%s_%s_%d.jpeg' % (meeting_n, iou_num, index)))
        print(join(middle_data_path, folder_name, cfg['base_conf']['mtcnn_origin_data_path'],
                   '%s_%s_%d.jpeg' % (meeting_n, iou_num, index)))
        index += 1

    f = open(join(middle_data_path, folder_name, '%d_%s.png' % (thres, tools.get_name_by_list(people_name))), 'w')
    f.close()
    start_time += section_time




