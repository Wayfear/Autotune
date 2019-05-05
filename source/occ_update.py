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


def main(args):
    load_label = bool(args.load_label)
    err_rate = args.err_rate
    cycle_num = args.cycle_num

    project_dir = os.path.dirname(os.getcwd())
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)

    thres = cfg['pre_process']['wifi_threshold']
    middle_data_path = join(project_dir, cfg['base_conf']['middle_data_path'])
    meeting_npy_paths = tools.get_meeting_and_path(middle_data_path, r'.+\.npy')
    middle_pic_path = tools.get_meeting_and_path(middle_data_path, r'.+\.pk$')
    cycle_people = tools.get_meeting_and_path(middle_data_path, r'.+\.pk' + re.escape(str(cycle_num-1)) + r'$')
    all_meeting_people_name = tools.get_meeting_people_name(middle_data_path, thres)
    meeting_paths = os.listdir(middle_data_path)
    for k, v in all_meeting_people_name.items():
        all_meeting_people_name[k] = [p for p in v if p not in cfg['filter_people']]

    true_label = {}
    if load_label:
        with open(join(project_dir, 'middle_data', 'true_label.pk'), 'rb') as f:
            true_label = pickle.load(f)
    else:
        for meet in meeting_paths:
            name_paths = os.listdir(join(middle_data_path, meet, 'classifier'))
            for name in name_paths:
                if os.path.isfile(join(middle_data_path, meet, 'classifier',name)):
                    continue
                for pic in os.listdir(join(middle_data_path, meet, 'classifier', name)):
                    true_label[pic] = name

    tem_label = {}
    for k, v in true_label.items():
        temp = k.split('_')
        tem_label["%s_%s_%s" % (temp[0], temp[1], temp[2])] = v
    true_label = tem_label

    peoples = []

    meeting_index = {}
    index = 0
    meeting_name = []
    center_file = args.center_file

    with open(join(project_dir, 'final_result', center_file), 'rb') as f:
        centers = pickle.load(f)

    for k, v in all_meeting_people_name.items():
        peoples.extend(v)
        meeting_index[k] = index
        index += 1
        meeting_name.append(k)
    peo_list = peoples.copy()
    peoples = list(set(peoples))
    eff_peos = []
    peo_count = {}
    for p in peoples:
        ct = peo_list.count(p)
        peo_count[p] = ct
        if ct>0 and p in centers:
            eff_peos.append(p)

    people_num = len(peoples)
    meeting_num = len(all_meeting_people_name)

    context_infor = np.zeros([meeting_num, people_num])

    for k, v in all_meeting_people_name.items():
        for name in v:
            context_infor[meeting_index[k], peoples.index(name)] = 1

    for k in middle_pic_path:
        with open(middle_pic_path[k], 'rb') as f:
            iou_pic_path = pickle.load(f)
        iou_vec = np.load(meeting_npy_paths[k])
        num = 3*len(all_meeting_people_name[k])

        people_dict = None
        if k in cycle_people:
            with open(cycle_people[k], 'rb') as f:
                people_dict = pickle.load(f)
                people_in_this_meeting = set(people_dict.keys())
        else:
            people_in_this_meeting = set(all_meeting_people_name[k])
        if len(iou_vec) < num:
            num = len(iou_vec)
        if num >= 3:
            clusters = AgglomerativeClustering(n_clusters=num, linkage='average').fit(iou_vec)
        else:
            continue
        cluster_feature = {}
        cluster_path = {}
        for cluster_num, feat, path in zip(clusters.labels_, iou_vec, iou_pic_path):
            if cluster_num not in cluster_feature:
                cluster_feature[cluster_num] = []
                cluster_path[cluster_num] = []
            cluster_feature[cluster_num].append(feat)
            cluster_path[cluster_num].append(path)
        save_matrix = {}
        for num in cluster_feature:
            save_matrix[num] = np.mean(np.array(cluster_feature[num]), axis=0)
        people = ''
        predict_peoples = {}
        for num in save_matrix:
            min_dis = 10000
            for p in eff_peos:
                dis = np.linalg.norm(save_matrix[num] - centers[p])
                if min_dis > dis:
                    min_dis = dis
                    people = p
            rate, true_pe = tools.get_cluster_belong_to_who(cluster_path[num], true_label)
            print('predict %s, true %s, rate %f, min_dis %f'%(people, true_pe, rate, min_dis))
            if people not in predict_peoples:
                predict_peoples[people] = min_dis
            else:
                if predict_peoples[people] > min_dis:
                    predict_peoples[people] = min_dis
        prediect_peo_list = list(predict_peoples.keys())
        predict_value_list = tools.dst2reliable(list(predict_peoples.values()))
        for i in range(len(prediect_peo_list)):
            predict_peoples[prediect_peo_list[i]] = predict_value_list[i]
        new_peops = list(people_in_this_meeting | set(prediect_peo_list))

        pre_vec = np.zeros(len(new_peops)).astype(np.float64)
        if people_dict is None:
            for i in range(len(new_peops)):
                if new_peops[i] in people_in_this_meeting:
                    pre_vec[i] = 1
        else:
            for i in range(len(new_peops)):
                if new_peops[i] in people_dict:
                    pre_vec[i] = people_dict[new_peops[i]]

        now_vec = np.array(len(new_peops)*[-1]).astype(np.float64)
        for i in range(len(new_peops)):
            if new_peops[i] not in eff_peos:
                now_vec[i] = 0
            if new_peops[i] in predict_peoples:
                now_vec[i] = predict_peoples[new_peops[i]]

        now_vec = tools.check_scale(pre_vec + err_rate*now_vec)
        result = dict(zip(new_peops, now_vec))
        result = {k: v for k, v in result.items() if v != 0}
        path = os.path.split(middle_pic_path[k])
        name = tools.get_name_by_dict(result)
        if len(name)>100:
            name = name[:100]
        with open(join(path[0], '%s.pk%d' % (name, cycle_num)), 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load_label', type=int,
                        help='Directory where to place model.', default=1)
    parser.add_argument('-e', '--err_rate', type=float,
                        help='The number of dimension in model result.', default=0.5)
    parser.add_argument('-c', '--cycle_num', type=int,
                        help='The number of dimension in model result.', default=1)
    parser.add_argument('-f', '--center_file', type=str,
                        help='center file store center.', default='0.05_49_center.pk')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    tools.email_subject(os.path.basename(__file__))

