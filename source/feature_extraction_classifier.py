from os.path import join
import argparse
import tensorflow as tf
import facenet
import os
import cv2
import numpy as np
from shutil import copyfile, rmtree
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import re
import tools
import yaml
import pickle
import sys


def main(args):

    project_dir = os.path.dirname(os.getcwd())
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)

    if cfg['specs']['set_gpu']:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['base_conf']['gpu_num'])

    middle_data_path = join(project_dir, cfg['base_conf']['middle_data_path'])
    thres = cfg['pre_process']['wifi_threshold']
    drop_path = tools.get_meeting_people_name(middle_data_path, thres)
    for k, v in drop_path.items():
        if len(v[0])==0:
            rmtree(join(middle_data_path, k))

    if True:
        for meeting in os.listdir(middle_data_path):
            temp_path = join(project_dir, cfg['base_conf']['middle_data_path'], meeting, 'classifier')
            if os.path.exists(temp_path):
                rmtree(temp_path)
        temp_path = tools.get_meeting_and_path(join(project_dir, cfg['base_conf']['middle_data_path']), r'.+\.npy$')
        for path in temp_path:
            if os.path.exists(temp_path[path]):
                os.remove(temp_path[path])

    meeting_npy_paths = tools.get_meeting_and_path(middle_data_path, r'.+\.npy$')
    meeting_paths = os.listdir(middle_data_path)

    meeting_paths = list(filter(lambda x: x not in meeting_npy_paths and x != 'time' and x != 'tmp' and x != 'true_label.pk',  meeting_paths))

    meeting_paths = list(map(lambda x: join(middle_data_path, x), meeting_paths))

    meeting_people_num = tools.get_meeting_people_num(middle_data_path, thres)

    for meeting, num in cfg['feature_extraction_classifier']['meeting_people_num'].items():
        if num > 0:
            meeting_people_num[str(meeting)] = num

    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(cfg['base_conf']['gpu_num'])
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            facenet.load_model(join(project_dir, 'models', args.model_dir))
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            for pic_path in meeting_paths:
                tmp_path = join(pic_path, cfg['base_conf']['mtcnn_origin_data_path'])
                print(tmp_path)
                pic_paths = os.listdir(tmp_path)
                pic_paths.sort(key=lambda x: int(re.split('\.|_', x)[2]))
                meeting = tools.get_parent_folder_name(pic_path, 1)
                image_list = []
                result = np.empty([0, args.dimension_num])
                piece_num = cfg['feature_extraction_classifier']['piece_num']
                if len(pic_paths) == 0:
                    rmtree(pic_path)
                    continue
                for pic in pic_paths:
                    im = cv2.imread(join(tmp_path, pic))
                    print(join(tmp_path, pic))
                    prewhitened = facenet.prewhiten(im)
                    image_list.append(prewhitened)
                    if len(image_list) == piece_num:
                        images = np.stack(image_list)
                        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        result = np.vstack((result, emb))
                        print(emb.shape)
                        image_list.clear()
                if len(image_list) != 0:
                    images = np.stack(image_list)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    result = np.vstack((result, emb))
                print(result.shape)

                features = result.tolist()

                pre_iou_num = int(re.split('\.|_', pic_paths[0])[2])
                same_iou_vec = np.empty([0, args.dimension_num])
                iou_vec = []
                iou_pic_path = []
                single_iou_path = []
                for name, vec in zip(pic_paths, features):
                    iou_num = int(re.split('\.|_', name)[2])
                    if iou_num == pre_iou_num:
                        same_iou_vec = np.vstack((same_iou_vec, vec))
                        single_iou_path.append(join(pic_path, 'mtcnn', name))
                    else:
                        iou_vec.append(np.mean(same_iou_vec, axis=0))
                        iou_pic_path.append(single_iou_path)
                        single_iou_path = []
                        same_iou_vec = np.empty([0, args.dimension_num])
                        same_iou_vec = np.vstack((same_iou_vec, vec))
                        single_iou_path.append(join(pic_path, 'mtcnn', name))
                        pre_iou_num = iou_num
                if len(single_iou_path) != 0:
                    iou_vec.append(np.mean(same_iou_vec, axis=0))
                    iou_pic_path.append(single_iou_path)

                with open(join(pic_path, 'pics%s.pk' % meeting), 'wb') as f:
                    pickle.dump(iou_pic_path, f)
                np.save(join(pic_path, 'vec%s.npy' % meeting), iou_vec)

                classifier_path = join(pic_path, 'classifier')
                if os.path.exists(classifier_path):
                    rmtree(classifier_path)
                os.mkdir(classifier_path)
                for i in range(meeting_people_num[str(meeting)]):
                    os.makedirs(join(classifier_path, str(i)))

                # print(iou_vec)
                if len(iou_vec) == 0:
                    continue

                if len(iou_vec) < 2:
                    continue
                if len(iou_vec) > meeting_people_num[meeting]:
                    cluster_number = meeting_people_num[meeting]
                else:
                    cluster_number = len(iou_vec)
                kmeans = AgglomerativeClustering(n_clusters=cluster_number, linkage='average').fit(iou_vec)
                print(kmeans.labels_.tolist())

                index = 0
                for d in kmeans.labels_:
                    # tools.copy_file_list('', iou_pic_path[index], join(pic_path, 'classifier', str(d)))
                    name = tools.get_parent_folder_name(iou_pic_path[index][0], 1)
                    copyfile(iou_pic_path[index][0], join(pic_path, 'classifier', str(d), name))
                    index += 1


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str,
                        help='Directory where to place model.', default='20180402-114759.pb')
    parser.add_argument('-d', '--dimension_num', type=int,
                        help='The number of dimension in model result.', default=512)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    tools.email_subject(os.path.basename(__file__))