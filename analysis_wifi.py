from source.tools import get_format_file, paser_wifi_file, plot_wifi_pic
import os
from os.path import join
import yaml
import numpy as np
import re
import source.tools as tools
from datetime import datetime
import shutil

project_dir = os.getcwd()
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

data_path = join(project_dir, 'wifi_data')

origin_data_path = join(project_dir, cfg['base_conf']['origin_data_path'])
wifi_files = get_format_file(data_path, 1, r'.+\.txt$')


for file in wifi_files:
    print(file)
    names = re.split(r'\/|\.|\\', file)
    name = names[-2]
    names = name.split('_')
    if names[0] == 'desktop':
        names = names[1:]
    start_time = datetime(year=int(names[0]), month=int(names[1]),
                                   day=int(names[2]), hour=int(names[3]), minute=int(names[4]))
    start_time = tools.get_nearby_time(1, start_time)

    start_time = datetime(year=2018, month=3, day=1, hour=1)

    for thres in range(-95, -24, 10):
        # start = datetime.now()
        # result1 = tools.parse_wifi_file_by_duration(15, file)
        # print(datetime.now() - start)
        # temp = datetime.now()
        # result2 = tools.parse_wifi_file_by_duration_multi_process(15, file)
        # print(datetime.now() - temp)
        # thres_result = tools.filter_wifi_result(result, thres)
        # tools.draw_wifi_distribution(thres_result, join(data_path, '%s_%d.png' % (name, thres)), cfg['mac_name'])
        tools.split_wifi_file_by_duration(start_time, 20, file, result_path=origin_data_path, folder=True)
        # _, sta = paser_wifi_file(file, thres, 600)
        # max_size = 0
        # for d in sta:
        #     s = list(sta[d].keys())
        #     if len(s) == 0:
        #         continue
        #     if max(s) > max_size:
        #         max_size = max(s)
        #
        # for d in sta:
        #     sta[d] = dict(filter(lambda x: x[1] > 2, sta[d].items()))
        #
        # y_lable = []
        # file_name = ''
        # sta = dict(filter(lambda x: len(x[1]) > 0, sta.items()))
        # arr = np.zeros(shape=[len(sta), max_size + 1])
        # index = 0
        # for d in sta:
        #     for m in sta[d]:
        #         arr[index, int(m)] = 200
        #     index += 1
        # for li in list(sta.keys()):
        #     y_lable.append(cfg['mac_name'][li])
        #     file_name += (cfg['mac_name'][li] + '_')
        # file_name = file_name[:-1]
        # wifi_file_name = re.split(r'\.|\/', file)[-2]
        # plot_wifi_pic(arr, y_lable,
        #                     join(data_path, '%s_%d_%s.png' % (wifi_file_name, thres, file_name)))

        # new_dst = join(data_path, 'tmp')
        # print('Move to new dst: {}'.format(join(new_dst, name + '.txt')))
        # shutil.move(file, join(new_dst, name + '.txt'))

