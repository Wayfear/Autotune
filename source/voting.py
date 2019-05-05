import os
import yaml
from os.path import join
import numpy as np
import re
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import pickle
import tools
import argparse
import sys
import csv
from shutil import copyfile, rmtree


def main(args):
    project_dir = os.path.dirname(os.getcwd())
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)

    middle_data_path = join(project_dir, cfg['base_conf']['middle_data_path'])
    middle_pic_path = tools.get_meeting_and_path(middle_data_path, r'.+\.pk$')
    meeting_npy_paths = tools.get_meeting_and_path(middle_data_path, r'.+\.npy$')
    result_data_path = join(project_dir, 'final_result')
    paths = os.listdir(result_data_path)
    paths = list(filter(lambda x: re.match(re.escape(args.header) + r'.+\.csv', x), paths))

    video_info = np.empty([0, args.dimension_num])
    pic_paths = []

    for k in middle_pic_path:
        with open(middle_pic_path[k], 'rb') as f:
            iou_pic_path = pickle.load(f)
            pic_paths.extend(iou_pic_path)
        iou_vec = np.load(meeting_npy_paths[k])
        try:
            video_info = np.vstack((video_info, iou_vec))
        except:
            print(k)

    stat = {k[0]: {} for k in pic_paths}
    tmp_paths = pic_paths

    pic_paths = [k[0] for k in pic_paths]

    vecters = dict(zip(pic_paths, video_info))

    peoples = set([])

    for path in paths:
        temp_path = os.path.split(path)[-1]
        temp_path = re.split(r'-|_', temp_path)
        if temp_path[-1][:-5] == 'voting':
            continue
        add_num = int(temp_path[-1][:-4])
        if args.start_num > add_num or args.end_num < add_num:
            continue
        with open(join(project_dir, 'final_result', path)) as f:
            result = csv.reader(f, delimiter=',')
            for row in result:
                if row[0] in stat:
                    peoples.add(row[1])
                    if row[1] in stat[row[0]]:
                        stat[row[0]][row[1]] += 1
                    else:
                        stat[row[0]][row[1]] = 1
    peoples = list(peoples)
    peoples.sort()
    soft_label = {k[0]: np.zeros([len(peoples)]).astype(np.float64) for k in tmp_paths}
    for k, v in stat.items():
        for p, n in v.items():
            soft_label[k][peoples.index(p)] = n

    for k, v in soft_label.items():
        if v.sum()==0:
            soft_label[k] = np.zeros([len(peoples)]).astype(np.float64)
        soft_label[k] = v/v.sum()
    save_soft_label = {}
    for k, v in soft_label.items():
        save_soft_label[tools.get_parent_folder_name(k, 1)] = v

    result = {k: find_max_key(v, 0) for k, v in stat.items()}

    centers = {}

    for k, v in result.items():
        if v is None:
            continue
        if v not in centers:
            centers[v] = np.empty([0, args.dimension_num])
        centers[v] = np.vstack((centers[v], vecters[k]))
    for k in centers:
        centers[k] = np.mean(centers[k], axis=0)
    with open(join(project_dir, 'final_result', '%s_voting_center.pk' % args.header), 'wb') as f:
        pickle.dump(centers, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(join(project_dir, 'final_result', '%s_%d_%d_voting.csv%d' % (args.header, args.start_num, args.end_num, args.flag)), 'w') as f:
        for k, v in result.items():
            if v is None:
                continue
            f.write('%s,%s\n' % (k, v))
    if os.path.exists(join(project_dir, 'final_result', '%s_%d_%d_voting' % (args.header, args.start_num, args.end_num))):
        rmtree(join(project_dir, 'final_result', '%s_%d_%d_voting' % (args.header, args.start_num, args.end_num)))
    os.mkdir(join(project_dir, 'final_result', '%s_%d_%d_voting' % (args.header, args.start_num, args.end_num)))

    index = 0
    for k, v in result.items():
        if v is None:
            continue
        if not os.path.exists( join(project_dir, 'final_result',
                                '%s_%d_%d_voting' % (args.header, args.start_num, args.end_num), v)):
            os.mkdir(join(project_dir, 'final_result', '%s_%d_%d_voting' % (args.header, args.start_num, args.end_num), v))

        s_pic_name = tools.get_parent_folder_name(k, 1)
        s_meet_name = tools.get_parent_folder_name(k, 3)
        copyfile(join(middle_data_path, s_meet_name, 'mtcnn', s_pic_name),
                 join(project_dir, 'final_result', '%s_%d_%d_voting' % (args.header, args.start_num, args.end_num),
                      v, s_pic_name))
        index += 1

    with open(join(project_dir, 'final_result', '%s_%d_%d_voting' % (args.header, args.start_num, args.end_num),
                   'soft_label.pk'), 'wb') as f:
        pickle.dump(save_soft_label, f, protocol=pickle.HIGHEST_PROTOCOL)


def find_max_key(dic, threshold=5):
    if len(dic)==0:
        return None
    ma = 0
    people = ''
    sum = 0
    for k, v in dic.items():
        sum+=v
        if v >= ma:
            ma = v
            people = k
    if sum > threshold:
        return people
    else:
        return None


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start_num', type=int,
                        help='The start of add people num.', default=20)
    parser.add_argument('-e', '--end_num', type=int,
                        help='The end of add people num.', default=50)
    parser.add_argument('-d', '--header', type=str,
                        help='The header name of result file.', default='')
    parser.add_argument('-n', '--dimension_num', type=int,
                        help='The number of dimension in model result.', default=512)
    parser.add_argument('-f', '--flag', type=int,
                        help='The number of dimension in model result.', default=1)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    tools.email_subject(os.path.basename(__file__))

