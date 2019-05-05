import os
import yaml
from os.path import join
from datetime import datetime
import csv
import pickle
import re
import argparse
import sys
from shutil import copyfile
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tools


def main(args):
    load_label = bool(args.load_label)

    project_dir = os.path.dirname(os.getcwd())
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)

    middle_data_path = join(project_dir, cfg['base_conf']['middle_data_path'])
    meeting_paths = os.listdir(middle_data_path)
    meeting_paths = list(filter(lambda x: x != 'time' and x != 'tmp' and x != 'problem data', meeting_paths))

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

        if os.path.isfile(join(project_dir, 'middle_data', 'true_label.pk')):
            time = datetime.strftime(datetime.now(), '%m-%d-%H-%M-%S')
            copyfile(join(project_dir, 'middle_data', 'true_label.pk'),
                     join(project_dir, 'important_file', '%s_true_label.pk'%time))
        with open(join(project_dir, 'middle_data', 'true_label.pk'), 'wb') as f:
            pickle.dump(true_label, f, protocol=pickle.HIGHEST_PROTOCOL)
    peoples = []
    for k, v in true_label.items():
        if v!='other' and v not in cfg['filter_people']:
            peoples.append(v)
    peoples = list(set(peoples))

    result_data_path = join(project_dir, 'final_result')
    paths = os.listdir(result_data_path)
    paths = list(filter(lambda x: re.match(re.escape(args.header) + r'.+\.csv', x), paths))
    # print(true_label)

    tem_label = {}
    for k,v in true_label.items():
        temp = k.split('_')
        tem_label["%s_%s_%s"%(temp[0], temp[1], temp[2])] = v
    true_label = tem_label
    max_person_sta = {}

    for p in peoples:
        max_person_sta[p] = 0

    draw_acc_x = []
    draw_acc_y = []
    with open(join(project_dir, 'report', '%s_report.csv'%args.header), 'w') as final_report:
        final_report.write('file_name,hyper_para,add_person,size,miss_num,accuarcy')
        for p in peoples:
            # print(cfg['mac_name'][p])
            final_report.write(',%s,%s_size'%(p,p))
        final_report.write(',mean_acc')
        final_report.write('\n')
        for path in paths:
            temp_path = os.path.split(path)[-1]
            final_report.write('%s,' % temp_path)
            temp_path = re.split(r'-|_', temp_path)
            final_report.write('%s,%s,'%(temp_path[-2], temp_path[-1][:-4]))
            if temp_path[-1][:-5] != 'voting':
                draw_acc_x.append(int(temp_path[-1][:-4]))
            else:
                draw_acc_x.append(0)
            true = 0
            miss = 0
            all = 0
            person_sta_true = {}
            person_sta_false = {}
            with open(join(project_dir, 'final_result', path)) as f:
                result = csv.reader(f, delimiter=',')
                for row in result:
                    r = row[0].split('/')
                    if row[1] not in person_sta_false:
                        person_sta_false[row[1]] = 0
                    if row[1] not in person_sta_true:
                        person_sta_true[row[1]] = 0
                    temp = r[-1].split('_')
                    r[-1] = '%s_%s_%s' % (temp[0], temp[1], temp[2])
                    if r[-1] in true_label:
                        all += 1
                        if true_label[r[-1]] == row[1]:
                            true += 1
                            person_sta_true[row[1]] += 1
                        else:
                            # print('label %s, predict %s'%(true_label[r[-1]], row[1]))
                            person_sta_false[row[1]] += 1
                    else:
                        miss += 1
                        print('miss pic %s'%r[-1])
                print("true total pic %d, miss pic %d"%(all, miss))
                final_report.write('%d,%d,' % (all, miss))
            if all == 0:
                acc = 0
            else:
                acc = true/all
            final_report.write('%f' % acc)
            print('acc %f'%acc)
            mean_acc = []
            for p in peoples:
                if p in person_sta_true:
                    if person_sta_true[p]+person_sta_false[p] == 0:
                        acc_per = 0
                    else:
                        acc_per = person_sta_true[p]/(person_sta_true[p]+person_sta_false[p])
                    print("%s : %f size: %d" % (p, acc_per, person_sta_true[p] + person_sta_false[p]))
                    mean_acc.append(acc_per)
                else:
                    acc_per = 0
                if acc_per>max_person_sta[p]:
                    max_person_sta[p] = acc_per

            draw_acc_y.append(sum(mean_acc) / float(len(mean_acc)))

            for p in peoples:
                # print(cfg['mac_name'][p])
                if p in person_sta_true:
                    if person_sta_true[p] + person_sta_false[p] == 0:
                        acc_per = 0
                    else:
                        acc_per = person_sta_true[p] / (person_sta_true[p] + person_sta_false[p])
                    final_report.write(',%f,%d' % (acc_per, person_sta_true[p] + person_sta_false[p]))
                else:
                    final_report.write(',0,0')
            print('\n')
            final_report.write(',%f'%draw_acc_y[-1])
            final_report.write('\n')

        for k, v in max_person_sta.items():
            print("%s max acc rate: %f"%(k, v))
        plt.figure(figsize=(12, 4))
        di = list(zip(draw_acc_x, draw_acc_y))
        di.sort(key=lambda x:x[0])
        plt.plot([a[0] for a in di], [a[1] for a in di], color="red", linewidth=2)
        plt.ylabel("Mean Acc(%)")
        plt.xlabel("Add Person")
        plt.title("Accuracy plot")
        plt.savefig(join(project_dir, 'report', '%s_report.jpg'%args.header))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load_label', type=int,
                        help='Directory where to place model.', default=1)
    parser.add_argument('-d', '--header', type=str,
                        help='The header name of result file.', default='2018-04-16-22-18-30')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    tools.email_subject(os.path.basename(__file__))