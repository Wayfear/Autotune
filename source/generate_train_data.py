import argparse
import sys
import os
from os.path import join
import itertools
import random
from sklearn.model_selection import train_test_split
from shutil import copyfile, rmtree
import tools


def main(args):
    project_dir = os.path.dirname(os.getcwd())
    source_data_dir = join(project_dir, 'final_result', args.source_data_dir)
    train = {}
    test = {}
    for d in os.listdir(source_data_dir):
        if os.path.isfile(join(source_data_dir, d)):
            continue
        pics = os.listdir(join(source_data_dir, d))
        pics = list(map(lambda x: join(source_data_dir, d, x), pics))
        tra, te = train_test_split(pics, test_size=0.2)
        train[d] = tra
        test[d] = te

    base_path = join(project_dir, 'fine_tuning_process', 'data')
    os.mkdir(join(base_path, args.data_dir))
    os.mkdir(join(base_path, args.data_dir, "train"))
    os.mkdir(join(base_path, args.data_dir, "test"))

    for t in train:
        os.mkdir(join(base_path, args.data_dir, "train", t))
        for p in train[t]:
            path = os.path.split(p)
            copyfile(p, join(base_path, args.data_dir, "train", os.path.split(path[-2])[-1], path[-1]))

    copyfile(join(source_data_dir, 'soft_label.pk'), join(base_path, args.data_dir, "train", 'soft_label.pk'))

    for t in test:
        os.mkdir(join(base_path, args.data_dir, "test", t))
        for p in test[t]:
            path = os.path.split(p)
            copyfile(p, join(base_path, args.data_dir, "test", os.path.split(path[-2])[-1], path[-1]))

    tmp_path = join(base_path, args.data_dir, 'test')
    people_paths = os.listdir(tmp_path)

    pics = {}
    for p in people_paths:
        if os.path.isdir(join(tmp_path, p)):
            pics[p] = os.listdir(join(tmp_path, p))
    pair_peo = list(itertools.combinations(pics.keys(), 2))
    classes = len(pics) + len(pair_peo)
    per_class = args.test_case_num//classes
    # result_file = os.path.expanduser(args.data_dir)
    with open(join(base_path, args.data_dir, 'pairs.txt'), "w") as f:
        balance = 0
        count = 0
        f.write(str(per_class*classes)+'\n')
        for p in people_paths:
            pic_pair = list(itertools.combinations(pics[p], 2))
            if len(pic_pair) < per_class - balance:
                tmp_per_class = len(pic_pair)
            else:
                tmp_per_class = per_class - balance
            for paths in random.sample(pic_pair, tmp_per_class):
                f.write("%s\t%s\t%s\n"%(p, paths[0], paths[1]))
            count += tmp_per_class
            balance -= (per_class - tmp_per_class)
        balance = 0
        for p in pair_peo:
            pic_pair = list(itertools.product(pics[p[0]], pics[p[1]]))
            if len(pic_pair) < per_class - balance:
                tmp_per_class = len(pic_pair)
            else:
                tmp_per_class = per_class - balance
            for paths in random.sample(pic_pair, tmp_per_class):
                f.write("%s\t%s\t%s\t%s\n"%(p[0], paths[0], p[1], paths[1]))
            count += tmp_per_class
            balance -= (per_class - tmp_per_class)
        gap = args.test_case_num - count
        while gap > 0:
            for p in people_paths:
                pic_pair = list(itertools.combinations(pics[p], 2))
                if len(pic_pair) < gap:
                    tmp_per_class = len(pic_pair)
                else:
                    tmp_per_class = gap
                for paths in random.sample(pic_pair, tmp_per_class):
                    f.write("%s\t%s\t%s\n" % (p, paths[0], paths[1]))
                gap -= tmp_per_class


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        help='Directory where to place validation data.', default='data_folder')
    # parser.add_argument('--result_file', type=str,
    #                    help='Directory where to place validation data.', default='pairs.txt')
    parser.add_argument('--source_data_dir', type=str,
                        help='Directory where to place validation data.', default='04-14-14-52-51_0.050000_51')
    parser.add_argument('--test_case_num', type=int,
                        help='Number of test cases to use for cross validation. Mainly used for testing.', default=500)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    tools.email_subject(os.path.basename(__file__))
