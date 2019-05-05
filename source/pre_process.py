import tools
import os
import yaml
from os.path import join
import tensorflow as tf
import numpy as np
import libs.align.detect_face as detect_face
from scipy import misc
from shutil import copyfile, rmtree


project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)
middle_data_path = join(project_dir, cfg['base_conf']['middle_data_path'], 'time')
origin_data_path = join(project_dir, cfg['base_conf']['origin_data_path'])
origin_video_paths = tools.get_meeting_and_path_list(origin_data_path, r'.+\.MP4')
origin_wifi_paths = tools.get_meeting_and_path(origin_data_path, r'.+\.txt')

if cfg['pre_process']['refresh']:
    for meeting in origin_video_paths:
        temp_path = join(project_dir, cfg['base_conf']['middle_data_path'], meeting)
        if os.path.exists(temp_path):
            rmtree(temp_path)

middle_data_paths = os.listdir(middle_data_path)
origin_video_paths = dict(filter(lambda x: x[0] not in middle_data_paths, origin_video_paths.items()))

# second filter of date
# date_to_process = cfg['pre_process']['date_to_process']
# if date_to_process != 0:
#     origin_video_paths = dict(filter(lambda x:  x[0].startswith(date_to_process), origin_video_paths.items()))

print(origin_video_paths.keys())

if len(origin_video_paths) > 0:
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(cfg['base_conf']['gpu_num'])
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess,
                join(project_dir, cfg['base_conf']['model_path'], cfg['pre_process']['mtcnn_model_path']))
    face_cascade = cv2.CascadeClassifier(join(project_dir, cfg['base_conf']['model_path'],
             cfg['pre_process']['opencv_model_path'], 'haarcascade_frontalface_default.xml'))

iou_index = 0
iou_before = []
for meeting, video_paths in origin_video_paths.items():
    meeting_middle_data_path = join(middle_data_path, meeting)
    os.mkdir(meeting_middle_data_path)
    os.mkdir(join(meeting_middle_data_path, cfg['base_conf']['mtcnn_origin_data_path']))
    cv2.destroyAllWindows()

    thres = cfg['pre_process']['wifi_threshold']
    for th in range(thres - 10, thres + 30, 10):
        try:
            _, sta = tools.paser_wifi_file(origin_wifi_paths[meeting], th, cfg['pre_process']['wifi_time_interval'])
            max_size = 0
            for d in sta:
                s = list(sta[d].keys())
                if len(s) == 0:
                    continue
                if max(s) > max_size:
                    max_size = max(s)

            for d in sta:
                sta[d] = dict(filter(lambda x: x[1] > 2, sta[d].items()))

            y_lable = []
            file_name = ''
            sta = dict(filter(lambda x: len(x[1]) > 0, sta.items()))
            arr = np.zeros(shape=[len(sta), max_size + 1])
            index = 0
            for d in sta:
                for m in sta[d]:
                    arr[index, int(m)] = 200
                index += 1
            for li in list(sta.keys()):
                y_lable.append(cfg['mac_name'][li])
                file_name += (cfg['mac_name'][li] + '_')
            file_name = file_name[:-1]
            tools.plot_wifi_pic(arr, y_lable,
                                join(meeting_middle_data_path, '%d_%s.png' % (th, file_name)))
            print('Finish wifi data file %s' % ('%d_%s.png' % (th, file_name)))
        except KeyError:
            print('Can not find wifi data file!')
        except RuntimeError('Invalid DISPLAY variable'):
            print('Cannot plot wifi... Remember add plt.switch_backend(\`agg\`) after import matplotlib.pyplot as plt ')

    index = 0
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        frame_num = 0
        false_num = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if frame is None:
                false_num += 1
            if false_num > 120:
                break
            frame_num += 1
            if ret:
                num_rows, num_cols = frame.shape[:2]
                img_size = np.asarray(frame.shape)[0:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if gray.ndim == 2:
                    img = tools.to_rgb(gray)

                    bounding_boxes, _ = detect_face.detect_face(img, cfg['pre_process']['mtcnn_mini_size'],
                                                                pnet, rnet, onet, cfg['pre_process']['mtcnn_threshold'],
                                                                cfg['pre_process']['mtcnn_factor'])
                    print(bounding_boxes)
                    img_list = []
                    temp_iou = []
                    if len(bounding_boxes) >= 1:
                        # for b in bounding_boxes:
                        #     cv2.rectangle(gray, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
                        margin = 44

                        for j in range(len(bounding_boxes)):
                            temp_iou_index = -1
                            det = np.squeeze(bounding_boxes[j, 0:4])
                            bb = np.zeros(5, dtype=np.int32)
                            bb[0] = np.maximum(det[0] - margin / 2, 0)
                            bb[1] = np.maximum(det[1] - margin / 2, 0)
                            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                            pics = face_cascade.detectMultiScale(cropped, 1.1, 5)
                            if len(pics) != 1:
                                continue
                            if len(iou_before)!=0:
                                for four_num in iou_before:
                                    print("nums: %d"%len(iou_before))
                                    print(four_num)
                                    print(bb)
                                    test_iou = tools.cal_iou(img_size, four_num, bb)
                                    print("iou_id: %d"%four_num[4])
                                    print("iou: %f"%test_iou)
                                    if test_iou > 0.85:
                                        temp_iou_index = four_num[4]
                                        break
                                if temp_iou_index == -1:
                                    iou_index += 1
                                    temp_iou_index = iou_index
                            else:
                                iou_index += 1
                                temp_iou_index = iou_index
                            bb[4] = temp_iou_index
                            temp_iou.append(bb)

                            aligned = misc.imresize(cropped, (cfg['pre_process']['image_size'],
                                                              cfg['pre_process']['image_size']), interp='bilinear')
                            # if cfg['pre_process']['show_pic']:
                            #     cv2.imshow('frame', aligned)
                            cv2.imwrite(join(meeting_middle_data_path, cfg['base_conf']['mtcnn_origin_data_path'],
                                             '%d-%d-%d.jpeg' % (index, frame_num, temp_iou_index)), aligned)
                            print("save the pic %d-%d-%d.jpeg" % (index, frame_num, temp_iou_index))
                            index += 1
                        iou_before = temp_iou
        print('Finish video %s' % video_path)

cap.release()
cv2.destroyAllWindows()
