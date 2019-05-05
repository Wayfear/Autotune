import tools
import os
import yaml
import numpy as np
import re
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from munkres import Munkres, print_matrix
import pickle
from datetime import datetime, timedelta
from shutil import copyfile, rmtree
from os.path import join
import argparse
import sys


def main(args):
    copy_pic = bool(args.copy_pic)
    cycle_num = args.cycle_num
    start_num = args.start_num
    end_num = args.end_num

    project_dir = os.path.dirname(os.getcwd())
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)

    thres = cfg['pre_process']['wifi_threshold']
    middle_data_path = join(project_dir, cfg['base_conf']['middle_data_path'])
    meeting_npy_paths = tools.get_meeting_and_path(middle_data_path, r'.+\.npy$')
    middle_pic_path = tools.get_meeting_and_path(middle_data_path, r'.+\.pk$')
    all_meeting_people_name = tools.get_meeting_people_name(middle_data_path, thres)
    cycle_people = tools.get_meeting_and_path(middle_data_path, r'.+\.pk' + re.escape(str(cycle_num)) + r'$')

    peoples = []

    for k, v in all_meeting_people_name.items():
        all_meeting_people_name[k] = [p for p in v if p not in cfg['filter_people']]

    for p in all_meeting_people_name.values():
        peoples.extend(p)
    peoples = list(set(peoples))
    peoples.sort()

    meeting_name = list(all_meeting_people_name.keys())
    meeting_name.sort()
    meeting_num = len(all_meeting_people_name)

    meeting_index = dict(zip(meeting_name, list(range(meeting_num))))

    people_num = len(peoples)

    cycle_people_dict = {}
    for k, v in cycle_people.items():
        with open(cycle_people[k], 'rb') as f:
            cycle_people_dict[k] = pickle.load(f)

    context_infor = np.zeros([meeting_num, people_num]).astype(np.float64)

    # NOT use corrected people info
    for k, v in all_meeting_people_name.items():
        for name in v:
            context_infor[meeting_index[k], peoples.index(name)] = 1

    # use corrected people info
    # if len(cycle_people_dict) == 0:
    #     for k, v in all_meeting_people_name.items():
    #         for name in v:
    #             context_infor[meeting_index[k], peoples.index(name)] = 1
    # else:
    #     for meeting_name, peoples_dict in cycle_people_dict.items():
    #         for people_name, v in peoples_dict.items():
    #             context_infor[meeting_index[meeting_name], peoples.index(people_name)] = v

    video_info = np.empty([0, args.dimension_num])
    pic_paths = []
    file_paths = list(middle_pic_path.keys())
    file_paths.sort()
    for k in file_paths:
        with open(middle_pic_path[k], 'rb') as f:
            iou_pic_path = pickle.load(f)
            pic_paths.extend(iou_pic_path)
        iou_vec = np.load(meeting_npy_paths[k])
        try:
            video_info = np.vstack((video_info, iou_vec))
        except:
            print(k)

    for para in [float(args.hyper_para)/100]:
        pic_features = []
        print('start translate')
        print(len(pic_paths))
        for feature, path in zip(video_info, pic_paths):
            parent = tools.get_parent_folder_name(path[0], 3)
            c = parent.split('_')
            try:
                li = list(map(lambda x: x * para, context_infor[meeting_index['%s_%s' % (c[0], c[1])]]))
                temp = np.concatenate((feature, li), axis=0)
            except:
                print('err')
            pic_features.append(temp)
        print('finish translate')

        for add_num in range(start_num, end_num):

            pic_people_in_meetings = np.zeros([people_num+add_num, meeting_num])
            wifi_people_in_meetings = np.zeros([people_num, meeting_num])

            if len(cycle_people_dict) == 0:
                for i in range(people_num):
                    for name in meeting_name:
                        if peoples[i] in all_meeting_people_name[name]:
                            wifi_people_in_meetings[i, meeting_index[name]] = 1
            else:
                for name in meeting_name:
                    if name in cycle_people_dict:
                        for i in range(people_num):
                            if peoples[i] in cycle_people_dict[name]:
                                wifi_people_in_meetings[i, meeting_index[name]] = cycle_people_dict[name][peoples[i]]

            print('start cluster')
            clusters = AgglomerativeClustering(n_clusters=people_num+add_num, linkage='average').fit(pic_features)
            print('finish cluster')

            cluster_feature = {}
            for cluster_num, feat in zip(clusters.labels_, video_info):
                if cluster_num not in cluster_feature:
                    cluster_feature[cluster_num] = []
                cluster_feature[cluster_num].append(feat)

            stat = {}
            for cluster_num, pic_path in zip(clusters.labels_, pic_paths):
                parent = tools.get_parent_folder_name(pic_path[0], 3)
                tem = parent.split('_')
                if cluster_num not in stat:
                    stat[cluster_num] = set([])
                stat[cluster_num].add('%s_%s' % (tem[0], tem[1]))


            res_stat = {}
            for num in stat:
                res_stat[num] = {}
                for n in stat[num]:
                    res_stat[num][n] = []
            for cluster_num, pic_path in zip(clusters.labels_, pic_paths):
                parent = tools.get_parent_folder_name(pic_path[0], 3)
                tem = parent.split('_')
                res_stat[cluster_num]['%s_%s' % (tem[0], tem[1])].append(pic_path)

            for peo_no, in_meetings in stat.items():
                for m in in_meetings:
                    pic_people_in_meetings[peo_no, meeting_index[m]] = 1

            cost_matrix = np.zeros([people_num+add_num, people_num+add_num])

            for i in range(people_num+add_num):
                for j in range(people_num):
                    cost_matrix[i, j] = np.linalg.norm(pic_people_in_meetings[i] - wifi_people_in_meetings[j])
            #
            # for i in range(people_num+add_num):
            #     print(cost_matrix[i, :])

            m = Munkres()
            indexes = m.compute(cost_matrix*100)
            # print_matrix(cost_matrix*100, msg='Lowest cost through this matrix:')

            save_matrix = {}
            final_res = {}
            for row, column in indexes:
                if column < people_num and row < people_num+add_num:
                    name = peoples[column]
                    final_res[name] = res_stat[row]
                    save_matrix[name] = np.mean(np.array(cluster_feature[row]), axis=0)

            with open(join(project_dir, 'final_result', '%d_%d_center.pk'%(int(para*100), add_num)), 'wb') as f:
                pickle.dump(save_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

            # time = datetime.strftime(datetime.now(), '%m-%d-%H-%M-%S')

            with open(join(project_dir, 'final_result', '%s_%d_%d.csv'%(args.header, int(para*100), add_num)), 'w') as f:
                for k, v in final_res.items():
                    for iou in v:
                        for paths in v[iou]:
                            # for path in paths:
                            f.write('%s,%s\n'%(paths[0], k))

            if copy_pic:
                if os.path.exists(join(project_dir, 'final_result', '%s_%d_%d' % (args.header, int(para*100), add_num))):
                    rmtree(join(project_dir, 'final_result', '%s_%d_%d' % (args.header, int(para*100), add_num)))
                os.mkdir(join(project_dir, 'final_result', '%s_%d_%d' % (args.header, int(para*100), add_num)))
                for k, v in final_res.items():
                    index = 0
                    os.mkdir(join(project_dir, 'final_result', '%s_%d_%d'%(args.header, int(para*100), add_num), k))
                    for iou in v:
                        for paths in v[iou]:
                            # for path in paths:
                            s_pic_name = tools.get_parent_folder_name(paths[0], 1)
                            s_meet_name = tools.get_parent_folder_name(paths[0], 3)
                            copyfile(join(middle_data_path, s_meet_name, 'mtcnn', s_pic_name),
                                     join(project_dir, 'final_result', '%s_%d_%d'%(args.header, int(para*100), add_num), k, '%s_%d.jpeg'%(k, index)))
                            index += 1
            else:
                if '%d_%d'%(int(para*100), add_num) == args.export:
                    if os.path.exists(join(project_dir, 'final_result', '%s_%d_%d' % (args.header, int(para*100), add_num))):
                        rmtree(join(project_dir, 'final_result', '%s_%d_%d' % (args.header, int(para*100), add_num)))
                    os.mkdir(join(project_dir, 'final_result', '%s_%d_%d' % (args.header, int(para*100), add_num)))
                    for k, v in final_res.items():
                        index = 0
                        os.mkdir(join(project_dir, 'final_result', '%s_%d_%d' % (args.header, int(para*100), add_num), k))
                        for iou in v:
                            for paths in v[iou]:
                                # for path in paths:
                                s_pic_name = tools.get_parent_folder_name(paths[0], 1)
                                s_meet_name = tools.get_parent_folder_name(paths[0], 3)
                                copyfile(join(middle_data_path, s_meet_name, 'mtcnn', s_pic_name),
                                         join(project_dir, 'final_result', '%s_%d_%d' % (args.header, int(para*100), add_num),
                                              k, '%s_%d.jpeg' % (k, index)))
                                index += 1


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--copy_pic', type=int,
                        help='Whether to copy picture.', default=0)
    parser.add_argument('-c', '--cycle_num', type=int,
                        help='The number of cycle.', default=1)
    parser.add_argument('-s', '--start_num', type=int,
                        help='The start of add people num.', default=0)
    parser.add_argument('-e', '--end_num', type=int,
                        help='The end of add people num.', default=50)
    parser.add_argument('-d', '--header', type=str,
                        help='The header name of result file.', default='')
    parser.add_argument('-x', '--export', type=str,
                        help='Wheb copy pic is False, select a result copy pics.', default='')
    parser.add_argument('-n', '--dimension_num', type=int,
                        help='The number of dimension in model result.', default=512)
    parser.add_argument('-a', '--hyper_para', type=int,
                        help='The number of dimension in model result.', default=5)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    tools.email_subject(os.path.basename(__file__))
