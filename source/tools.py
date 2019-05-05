import os
import re
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import datetime
from datetime import datetime, timedelta
from matplotlib import cm
from numpy import arange
import pandas as pd
from os.path import join
import multiprocessing
from shutil import copyfile
import smtplib
import yaml
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools
from pandas import DataFrame
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pickle

def get_format_file(root_path, piles_num, pattern):
    paths = [root_path]
    for i in range(piles_num):
        paths = list(filter(lambda x: not os.path.isfile(x), paths))
        num = 0
        te = paths.copy()
        for path in te:
            temp = os.listdir(path)
            temp = list(map(lambda x: os.path.join(path, x), temp))
            paths.extend(temp)
            num += 1
        for k in range(num):
            paths.pop(0)
    paths = list(filter(lambda x: os.path.isfile(x) and re.match(pattern, x), paths))
    return paths


def get_parent_folder_name(direction, num=2):
    for i in range(num):
        path = os.path.split(direction)
        direction = path[0]
        result = path[1]
    return result


def get_meeting_and_path(path, pattern):
    paths = get_format_file(path, 2, pattern)
    result = {}
    for path in paths:
        result[get_parent_folder_name(path, 2)] = path
    return result


def get_meeting_and_path_list(path, pattern):
    paths = get_format_file(path, 2, pattern)
    result = {}
    for path in paths:
        parent = get_parent_folder_name(path, 2)
        if parent not in result:
            result[parent] = []
        result[parent].append(path)
    return result


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def paser_wifi_file(path, threshold, interval):
    result = {}
    statistics = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        bl = 0
        is_start = True
        for line in lines:
            line = line.strip('\n')
            l = line.split('\t')
            if len(l) == 2 or l[2] == '':
                continue
            try:
                strength = int(l[2])
            except:
                nums = l[2].split(',')
                strength = max(nums, key=lambda x: int(x))
                strength = int(strength)
            if strength < threshold:
                continue
            if l[1] not in statistics:
                statistics[l[1]] = {}
            time = l[0][:-3]
            try:
                time = datetime.strptime(time, '%b %d, %Y %H:%M:%S.%f')
            except:
                time = l[0][:-7]
                time = datetime.strptime(time, '%b %d, %Y %H:%M:%S.%f')
            if is_start:
                start_time = time
                is_start = False
                last_time = start_time
            t = (time - last_time).total_seconds()
            if t < interval:
                if bl not in statistics[l[1]]:
                    statistics[l[1]][bl] = 1
                else:
                    statistics[l[1]][bl] += 1
            else:
                bl += int(t / interval)
                last_time += timedelta(seconds=int(t / interval) * interval)
            if l[1] in result:
                result[l[1]].append({time: strength})
            else:
                result[l[1]] = [{time: strength}]
    return result, statistics


def plot_wifi_pic(data, y_classes, pic_name):
    fig, ax = plt.subplots()
    fig.set_size_inches(100, 20)
    ax.imshow(data, interpolation='nearest', cmap=cm.Blues)
    tick_marks_y = arange(len(y_classes))
    plt.yticks(tick_marks_y, y_classes)
    plt.ylabel('MAC Address')
    plt.xlabel('Time')
    plt.savefig(pic_name)


def read_mac_list(path):
    mac_address = pd.read_csv(path, header=None)
    mac_name = {}
    for k, v in zip(mac_address.ix[:, 0], mac_address.ix[:, 1]):
        mac_name[v] = k
    return mac_name


def get_meeting_people_num(path, thres):
    wifi_info_paths = get_format_file(path, 2, r'.+' + re.escape(str(thres)) + r'.+\.png$')
    result = {}
    for wifi in wifi_info_paths:
        name = get_parent_folder_name(wifi, 1)
        meeting = get_parent_folder_name(wifi, 2)
        result[meeting] = len(name.split('_')) - 1
    return result


def get_meeting_people_name(path, thres):
    wifi_info_paths = get_format_file(path, 2, r'.+' + re.escape(str(thres)) + r'.+\.png$')
    result = {}
    for wifi in wifi_info_paths:
        name = get_parent_folder_name(wifi, 1)
        meeting = get_parent_folder_name(wifi, 2)
        result[meeting] = re.split('_|\.|', name)[1:-1]

    return result


def paser_result_file(path):
    result = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            if len(line) < 1:
                continue
            l = line.split(': ')

            # temp = l[1].split('_')
            # result['%s_%s_%s' % (temp[-4], temp[-3], temp[-1])] = l[0]
            result[l[1]] = l[0]
    return result


def simple_paser_result_file(path):
    result = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            if len(line) < 1:
                continue
            l = line.split(': ')
            temp = l[1].split('_')
            if l[0] not in result:
                result[l[0]] = set([])
            result[l[0]].add('%s_%s' % (temp[-3], temp[-2]))
    return result


def parse_wifi_line(line):
    line = line.strip('\n')
    l = line.split('\t')
    if len(l) < 3 or l[2] == '':
        return []
    try:
        strength = int(l[2])
    except:
        nums = l[2].split(',')
        strength = max(nums, key=lambda x: int(x))
        strength = int(strength)
    time = l[0][:-3]
    try:
        time = datetime.strptime(time, '%b %d, %Y %H:%M:%S.%f')
    except:
        time = l[0][:-7]
        time = datetime.strptime(time, '%b %d, %Y %H:%M:%S.%f')
    if len(l[1]) != 17:
        return []
    return [time, l[1], strength]


def split_wifi_file_by_duration(start_time, minutes, path, result_path=None, folder=False):
    duration = timedelta(minutes=minutes)
    if result_path is None:
        result_path = path.split('.')[0]
        if not os.path.exists(result_path):
            os.mkdir(result_path)
    temp_start = start_time
    temp_end = start_time + duration
    file_name = '%s_%s' % (temp_start.strftime("%m-%d-%H-%M-%S"),
                                         temp_end.strftime("%m-%d-%H-%M-%S"))
    if folder:
        if not os.path.exists(join(result_path, file_name)):
            os.mkdir(join(result_path, file_name))
        split_f = open(join(result_path, file_name, '%s.txt' % file_name), 'w')
    else:
        split_f = open(join(result_path, '%s.txt' % file_name), 'w')
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_result = parse_wifi_line(line)
            if len(line_result) == 0:
                continue
            if line_result[0] > temp_start:
                while True:
                    if line_result[0] <= temp_end:
                        if split_f is None:
                            file_name = '%s_%s' % (temp_start.strftime("%m-%d-%H-%M-%S"),
                                                   temp_end.strftime("%m-%d-%H-%M-%S"))
                            if folder:
                                if not os.path.exists(join(result_path, file_name)):
                                    os.mkdir(join(result_path, file_name))
                                split_f = open(join(result_path, file_name, '%s.txt' % file_name), 'w')
                            else:
                                split_f = open(join(result_path, '%s.txt' % file_name), 'w')
                        split_f.write('%s000\t%s\t%d\n' % (line_result[0].strftime('%b %d, %Y %H:%M:%S.%f'), line_result[1], line_result[2]))
                        break
                    else:

                        split_f.close()
                        split_f = None
                        while line_result[0] >= temp_end:
                            temp_start = temp_end
                            temp_end = temp_start + duration


def get_nearby_time(minutes, solve_time):
    times = solve_time.minute // minutes
    return datetime(year=solve_time.year, month=solve_time.month,
                             day=solve_time.day,hour= solve_time.hour, minute=times * minutes)


def parse_wifi_file_by_duration(minutes, path):
    duration = timedelta(minutes=minutes)
    names = re.split(r'\/|\.', path)[-2]
    names = names.split('_')
    if names[0] == 'desktop':
        names = names[1:]
    if len(names) == 2:
        names = names[0].split('-')
        names.insert(0, '2017')
    print(names)
    start_time = datetime(year=int(names[0]), month=int(names[1]),
                                   day=int(names[2]), hour=int(names[3]), minute=int(names[4]))
    start_time = get_nearby_time(minutes, start_time)
    end_time = start_time + duration
    result = []
    with open(path, 'r') as f:
        lines = f.readlines()
        temp_record = {start_time: {}}
        for line in lines:
            line_result = parse_wifi_line(line)
            if len(line_result) == 0:
                continue
            if line_result[0] > start_time:
                while True:
                    if line_result[0] <= end_time:
                        if line_result[1] not in temp_record[start_time]:
                            temp_record[start_time][line_result[1]] = []
                        temp_record[start_time][line_result[1]].append(int(line_result[2]))
                        break
                    else:
                        result.append(temp_record)
                        start_time = end_time
                        end_time = start_time + duration
                        temp_record = {start_time: {}}
    return result


def filter_wifi_result(wifi_result, threshold):
    for times in wifi_result:
        for key, value in times.items():
            for mac, strengths in value.items():
                value[mac] = list(filter(lambda x: x > threshold, strengths))
            times[key] = dict(filter(lambda x: len(x[1]) > 0, value.items()))
    return wifi_result


def draw_wifi_distribution(filter_result, output_path, mac_name, color=cm.Blues):
    x_labels = []
    y_labels = set([])
    for times_bags in filter_result:
        for key, value in times_bags.items():
            x_labels.append(key)
            for mac in value:
                y_labels.add(mac)
    row = len(y_labels)
    y_labels = list(y_labels)
    x_labels = list(map(lambda x: x.strftime("%H:%M"), x_labels))
    column = len(x_labels)
    arr = np.zeros([row, column])
    for i in range(len(filter_result)):
        for key, value in filter_result[i].items():
            for mac, strengths in value.items():
                arr[y_labels.index(mac), i] = len(strengths) + 1000
    fig, ax = plt.subplots()
    fig.set_size_inches(100, 20)
    ax.imshow(arr, interpolation='nearest', cmap=color)
    y_labels = list(map(lambda x: mac_name[x], y_labels))
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position("right")
    # ax.set_yticklabels(y_labels)
    tick_marks_x = arange(-0.5, len(x_labels)-1, 1)
    plt.xticks(tick_marks_x, x_labels)
    tick_marks_y = arange(len(y_labels))

    plt.yticks(tick_marks_y, y_labels)
    plt.ylabel('Name')
    plt.xlabel('Time')
    plt.grid()

    plt.savefig(output_path)
    return y_labels


def get_name_by_list(names, separate='_'):
    name = ''
    for n in names:
        name += (n + separate)
    return name[:-1]


def get_name_by_dict(names, separate='_'):
    name = ''
    k_name = list(names.keys())
    k_name.sort()
    for k in k_name:
        name += '%s_%1.1f_' % (k, names[k])
    return name[:-1]


def worker(lines):
    result = []
    for line in lines:
        re = parse_wifi_line(line)
        if len(re) == 3:
            result.append(re)
    return result


def parse_wifi_file_by_duration_multi_process(minutes, path, numthreads=8, numlines=100):
    lines = open(path).readlines()
    num_cpu_avail = multiprocessing.cpu_count()
    numthreads = min(num_cpu_avail, numthreads)
    pool = multiprocessing.Pool(processes=numthreads)
    result_list = pool.map(worker,
        (lines[line:line+numlines] for line in range(0, len(lines), numlines)))
    pool.close()
    pool.join()
    result = []
    for re in result_list:
        result.extend(re)
    result.sort(key=lambda x: x[0])
    if len(result) == 0:
        return []

    duration = timedelta(minutes=minutes)
    start_time = get_nearby_time(minutes, result[0][0])
    end_time = start_time + duration
    final_result = []
    temp_record = {start_time: {}}
    for line_result in result:
        if line_result[0] > start_time:
            while True:
                if line_result[0] <= end_time:
                    if line_result[1] not in temp_record[start_time]:
                        temp_record[start_time][line_result[1]] = []
                    temp_record[start_time][line_result[1]].append(int(line_result[2]))
                    break
                else:
                    final_result.append(temp_record)
                    start_time = end_time
                    end_time = start_time + duration
                    temp_record = {start_time: {}}
    return final_result


def cal_iou(img_size, before, now):
    img = np.zeros([img_size[0], img_size[1]])
    img[before[1]:before[3], before[0]:before[2]] = 1
    img[now[1]:now[3], now[0]:now[2]] += 1
    count = 0
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            if img[i, j] == 2:
                count += 1
    print("count; %d"%count)
    return count/((before[3]-before[1])*(before[2]-before[0])+(now[3]-now[1])*(now[2]-now[0])-count)


def copy_file_list(file_folder, file_list, folder):
    for file in file_list:
        file_name = get_parent_folder_name(file, 1)
        copyfile(join(file_folder, file), join(folder, file_name))


def get_cluster_belong_to_who(paths, true_label):
    li = []
    miss = 0
    for path in paths:
        path = get_parent_folder_name(path[0], 1)
        temp = path.split('_')
        temp = '%s_%s_%s' % (temp[0], temp[1], temp[2])
        if temp in true_label:
            li.append(true_label[temp])
        else:
            miss += 1
            print('miss pic %s' % path)
    t_li = li.copy()
    li = list(set(li))
    max = -1
    ma = 0
    peo=''
    if len(t_li) == 0:
        return 0, ''
    for l in li:
        if t_li.count(l)>max:
            ma = t_li.count(l)
            peo = l
    return ma/len(t_li), peo


def dst2reliable(x, min_dst=0.4, max_dst=0.95):
    sig = 1/((max_dst-min_dst)**2)
    for i in range(len(x)):
        if x[i] <= min_dst:
            x[i] = 1
        elif x[i] >= max_dst:
            x[i] = 0
        else:
            x[i] = 1.0 - sig*((x[i]-min_dst)**2)
    return x


def check_scale(x, min=0, max=1):
    for i in range(len(x)):
        if x[i] < min:
            x[i] = 0
        elif x[i] > max:
            x[i] = 1
    return x


def email_subject(script_name):
    project_dir = os.path.dirname(os.getcwd())

    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)

    if cfg['specs']['send_email']:
        gmail_user = cfg['specs']['sender']
        gmail_password = 'sensornetwork'

        sent_from = gmail_user

        receiver = cfg['specs']['receiver']
        to = [receiver]

        subject = script_name
        msg = 'Subject:{}\n\n Successful!'.format(subject)

        try:
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.ehlo()
            server.login(gmail_user, gmail_password)
            server.sendmail(sent_from, to, msg)
            server.close()

        except:
            print('Something went wrong...')
            exit()
    else:
        return 0


def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap != None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          filename='../eval.pdf',
                          cmap=plt.cm.Blues,
                          figsize=(22, 20), ft_size=25, tick_size=30, label_size=35):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm[~np.any(np.isnan(cm), axis=1)]
        # print(cnf_matrix)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.rcParams.update({'font.size': ft_size})
    plt.rcParams.update({'figure.autolayout': True})
    plt.rc('xtick', labelsize=tick_size)
    plt.rc('ytick', labelsize=tick_size)

    plt.figure(figsize=figsize)
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True ID', fontsize=label_size)
    plt.xlabel('Predicted ID', fontsize=label_size)

    plt.savefig(filename)


def pie_error_type_folders(all_folders, err_type):
    return {
        0: [folder for folder in all_folders if float(folder.split('_')[0]) > 30 and float(folder.split('_')[3]) == 0],
        1: [folder for folder in all_folders if float(folder.split('_')[1]) > 0 and float(folder.split('_')[2]) == 0],
        2: [folder for folder in all_folders if float(folder.split('_')[2]) > 0 and float(folder.split('_')[1]) == 0],
        3:  [folder for folder in all_folders if float(folder.split('_')[3]) > 0],
        4: [folder for folder in all_folders if (float(folder.split('_')[1]) > 0 and float(folder.split('_')[2]) > 0)],
    }[err_type]


def load_pie_report(project_dir, cfg, method):
    # load data
    report_base = join(project_dir, cfg['draw']['errors']['report'], cfg['draw']['errors']['site'], method)
    all_folders = os.listdir(report_base)
    err_type = cfg['draw']['errors']['err_type']
    target_folders = pie_error_type_folders(all_folders, err_type)
    target_folders.sort()

    df = DataFrame(
            columns=('ctrl_var', 'accuracy', 'converge_iter.'))
    j = 1
    for folder in target_folders:
        try:
            ctrl_var = float(folder.split('_')[cfg['draw']['errors']['err_type']])
        except:
            ctrl_var = j
            j += 1
        csv_files = [f for f in os.listdir(join(report_base, folder)) if f.endswith('.csv')]
        csv_files.sort()
        i = 0
        tmp_acc = []
        for file_name in csv_files:
            csv_path = join(report_base, folder, file_name)
            with open(csv_path, "r") as filestream:
                key_line = [l for l in filestream if 'voting' in l]
                # print('folder: {}, file: {} has acc: {}'.format(folder, key_line[0].rstrip().split(",")[0], key_line[0].rstrip().split(",")[-1]))
            tmp_acc.append(float(key_line[0].rstrip().split(",")[-1]))
            i += 1
            # print(tmp_acc)
            if i > 2:
                window = np.array(tmp_acc[i-3:i])

                if window.std() < cfg['draw']['errors']['thres']:
                    # print('iter {} has std {}'.format(i, window.std()))
                    df.loc[len(df)] = [ctrl_var, tmp_acc[-1], i]
                    # print('folder {} converge!, df is {}'.format(folder, df))
                    break
                elif i == len(csv_files):
                    print('folder {} does NOT converge!'.format(folder))
                    df.loc[len(df)] = [ctrl_var, tmp_acc[-1], i]

    df = df.sort_values(by=['ctrl_var'])
    print('Method is {}'.format(method))
    print(df)

    return df

# use f1 instead of accuracy
def load_pie_report_v2(project_dir, cfg, method):
    # load data
    report_base = join(project_dir, cfg['draw']['errors']['report'], cfg['draw']['errors']['site'], method)
    all_folders = os.listdir(report_base)
    err_type = cfg['draw']['errors']['err_type']
    target_folders = pie_error_type_folders(all_folders, err_type)
    target_folders.sort()
    print(target_folders)

    # load true labels from pickle
    truth_dict = {}
    true_label_base = join(project_dir, cfg['draw']['errors']['report'], cfg['draw']['errors']['site'], cfg['draw']['errors']['truth_folder'])
    true_folders = os.listdir(true_label_base)
    for folder in true_folders:
        nb_non_poi = folder.split('_')[-1]

        with open(join(true_label_base, folder, 'true_label.pk'),
                  'rb') as f:
            raw_truth = pickle.load(f)
        truth = {}
        for k, v in raw_truth.items():
            temp = k.split('_')
            truth["%s_%s_%s" % (temp[0], temp[1], temp[2])] = v

        truth_dict[folder] = truth

    df = DataFrame(
            columns=('ctrl_var', 'precision', 'recall', 'F1 score', 'accuracy', 'converge_iter.'))
    j = 1
    for folder in target_folders:
        try:
            ctrl_var = float(folder.split('_')[cfg['draw']['errors']['err_type']])
        except:
            ctrl_var = j
            j += 1

        csv_files = list(filter(lambda x: re.match(r'.+voting\.csv', x), os.listdir(join(report_base, folder))))
        csv_files.sort()

        tmp_acc, tmp_precision, tmp_recall, tmp_f1 = [], [], [], []
        i = 0
        for file_name in csv_files:
            truth = truth_dict[folder]
            csv_path = join(report_base, folder, file_name)
            pred = {}

            with open(csv_path, "r") as filestream:
                for line in filestream:
                    currentline = line.rstrip().split(",")
                    temp = currentline[0].split('/')[-1].split('_')
                    pred["%s_%s_%s" % (temp[0], temp[1], temp[2])] = currentline[1]

            # extract the list of keys
            keys_truth = set(truth.keys())
            keys_pred = set(pred.keys())
            intersection = keys_truth & keys_pred
            print(intersection)
            # print('Size of intersection: {}'.format(len(intersection)))
            truth = {k: truth.get(k, None) for k in intersection}
            pred = {k: pred.get(k, None) for k in intersection}

            names = list(set().union(truth.values(), pred.values()))

            y_true = [truth[k] for k in intersection]
            y_pred = [pred[k] for k in intersection]

            # Compute confusion matrix
            report_lr = precision_recall_fscore_support(y_true, y_pred, average='macro')
            acc = accuracy_score(y_true, y_pred, normalize=True)
            print(
                "File %s: num_truth= %d, num_pred = %d, intersection = %d, precision = %0.3f, recall = %0.3f, F1 = %0.3f, accuracy = %0.3f\n" % \
                (csv_path.split('/')[-1], len(truth), len(pred), len(intersection), report_lr[0], report_lr[1],
                 report_lr[2],
                 acc))
            tmp_acc.append(acc)
            tmp_precision.append(report_lr[0])
            tmp_recall.append(report_lr[1])
            tmp_f1.append(report_lr[2])

            i += 1

            # check converge
            if i > 2:
                window = np.array(tmp_acc[i - 3:i])

                if window.std() < cfg['draw']['errors']['thres'][cfg['draw']['errors']['err_type']]:
                    # print('iter {} has std {}'.format(i, window.std()))
                    df.loc[len(df)] = [ctrl_var, tmp_precision[-1], tmp_recall[-1], tmp_f1[-1], tmp_acc[-1], i]
                    print('folder {} converge!'.format(folder))
                    break
                elif i == len(csv_files):
                    print('folder {} does NOT converge!'.format(folder))
                    df.loc[len(df)] = [ctrl_var, tmp_precision[-1], tmp_recall[-1], tmp_f1[-1], tmp_acc[-1], i]


    df = df.sort_values(by=['ctrl_var'])
    print('Method is {}'.format(method))
    print(df)

    return df


def scanfolder(parent_dir, postfix):
    target_files = []
    for path, dirs, files in os.walk(parent_dir):
        for f in files:
            if f.endswith(postfix):
                target_files.append(join(path, f))

    return target_files


def top_k_guesses(logits, ground_truth, num_chances):
    prediction_top_k = prob2decision(logits, num_chances)
    correct_prediction_counter = 0
    for truth, guesses in zip(ground_truth, prediction_top_k):
        if truth in guesses:
            correct_prediction_counter += 1
    refined_acc = np.array(correct_prediction_counter, dtype=np.float32) / ground_truth.shape[0]

    return refined_acc


def prob2decision(pred_prob, top_k):
    pred_top_k = []
    for instance in pred_prob:
        tmp = [i[0] for i in sorted(enumerate(instance), reverse=True, key=lambda x: x[1])]
        pred_top_k.append(tmp[:top_k])

    return np.array(pred_top_k)