from datetime import datetime
import os.path
import time
import sys
import random
import os
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import lfw
import h5py
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import gfile
import pickle
import tools

from os.path import join
import yaml


def main(args):
    project_dir = os.path.dirname(os.getcwd())
    network = importlib.import_module(args.model_def)

    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)

    if cfg['specs']['set_gpu']:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['base_conf']['gpu_num'])

    subdir = '%s_center_loss_factor_%1.2f' % (args.data_dir, args.center_loss_factor)

    # test = os.path.expanduser(args.logs_base_dir)
    log_dir = os.path.join(project_dir, 'fine_tuning_process', 'logs', subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(project_dir, 'fine_tuning_process', 'models', subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    data_dir = os.path.join(project_dir, 'fine_tuning_process', 'data', args.data_dir, 'train')

    train_set = facenet.get_dataset(data_dir)
    if args.filter_filename:
        train_set = filter_dataset(train_set, os.path.expanduser(args.filter_filename),
                                   args.filter_percentile, args.filter_min_nrof_images_per_class)
    nrof_classes = len(train_set)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = os.path.join(project_dir, 'fine_tuning_process', 'models', cfg['model_map'][args.embedding_size])  
    print('Pre-trained model: %s' % pretrained_model)
  
    lfw_dir = os.path.join(project_dir, 'fine_tuning_process', 'data', args.data_dir, 'test')
    print('LFW directory: %s' % lfw_dir)
    # Read the file containing the pairs used for testing
    lfw_pairs = os.path.join(project_dir, 'fine_tuning_process', 'data', args.data_dir, 'pairs.txt')
    pairs = lfw.read_pairs(lfw_pairs)
    # Get the paths for the corresponding images
    lfw_paths, actual_issame = lfw.get_paths_personal(lfw_dir, pairs)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # get soft labels
        with open(join(data_dir, 'soft_label.pk'), 'rb') as f:
            confidence_score = pickle.load(f)
        image_list, soft_labels_list = facenet.get_image_paths_and_soft_labels(train_set, confidence_score)
        soft_labels_array = np.array(soft_labels_list)
        soft_labels = ops.convert_to_tensor(soft_labels_array, dtype=tf.float32)

        assert len(image_list) > 0, 'The dataset should not be empty'

        # Create a queue that produces indices into the image_list and label_list
        range_size = array_ops.shape(soft_labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                                                    shuffle=True, seed=None, capacity=32)

        index_dequeue_op = index_queue.dequeue_many(args.batch_size * args.epoch_size, 'index_dequeue')

        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')

        hard_labels_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='hard_labels')

        soft_labels_placeholder = tf.placeholder(tf.float32, shape=(None, nrof_classes), name='soft_labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.float32],
                                              shapes=[(1,), (nrof_classes,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, soft_labels_placeholder],
                                              name='enqueue_op')

        nrof_preprocess_threads = 4
        images_and_softlabels = []
        for _ in range(nrof_preprocess_threads):
            filenames, soft_labels = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                if args.random_rotate:
                    image = tf.py_func(facenet.random_rotate_image, [image], tf.uint8)
                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)

                # pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))

            images_and_softlabels.append([images, soft_labels])

        image_batch, soft_label_batch = tf.train.batch_join(
            images_and_softlabels, batch_size=batch_size_placeholder)
        image_batch = tf.squeeze(image_batch, 1)

        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        soft_label_batch = tf.identity(soft_label_batch, 'soft_label_batch')

        print('Total number of classes: %d' % nrof_classes)
        print('Total number of examples: %d' % len(image_list))

        print('Building training graph')

        # Build the inference graph
        prelogits, _ = network.inference(image_batch, args.keep_probability,
                                         phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                                         weight_decay=args.weight_decay)
        # fine_tuning = slim.fully_connected(prelogits, args.embedding_size, activation_fn=None,
        #                            scope='FineTuning', reuse=False, trainable=True)

        logits = slim.fully_connected(prelogits, nrof_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(args.weight_decay),
                                      scope='Logits', reuse=False)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Add center loss
        if args.center_loss_factor > 0.0:
            prelogits_center_loss, _ = facenet.fuzzy_center_loss(prelogits, soft_label_batch,
                                                                 args.center_loss_alfa, args.fuzzier, nrof_classes)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)
            tf.summary.scalar('prelogits_center_loss', prelogits_center_loss)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.learning_rate_decay_epochs * args.epoch_size,
                                                   args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=soft_label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, args.optimizer,
                                 learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)

        # Create a saver
        all_vars = tf.trainable_variables()
        var_to_restore = [v for v in all_vars if not v.name.startswith('Logits')]
        saver = tf.train.Saver(var_to_restore, max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            if args.pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)
                result = sess.graph.get_tensor_by_name("InceptionResnetV1/Bottleneck/weights:0")
                pre = sess.graph.get_tensor_by_name("InceptionResnetV1/Block8/Branch_1/Conv2d_0c_3x1/weights:0")
                # tf.stop_gradient(persisted_result)
                # print(result.eval())
                # print("======")
                # print(pre.eval())

            # Training and validation loop
            print('Running training')
            epoch = 0
            pre_acc = -1
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, epoch, image_list, soft_labels_array, index_dequeue_op, enqueue_op,
                      image_paths_placeholder, soft_labels_placeholder,
                      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
                      total_loss, train_op, summary_op, summary_writer, regularization_losses,
                      args.learning_rate_schedule_file, logits)
                # print(result.eval())
                # print("======")
                # print(pre.eval())

                # Save variables and the metagraph if it doesn't exist already
                # Evaluate on LFW
                if args.lfw_dir:
                    acc = evaluate(sess, enqueue_op, image_paths_placeholder, soft_labels_placeholder, phase_train_placeholder,
                             batch_size_placeholder,
                             embeddings, soft_label_batch, lfw_paths, actual_issame, args.lfw_batch_size,
                             args.lfw_nrof_folds, log_dir, step, summary_writer, nrof_classes, prelogits_center_loss)
                    if acc > pre_acc:
                        save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)
                        pre_acc = acc
    return model_dir


def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile * 0.01, cdf, bin_centers)
    return threshold


def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename, 'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center >= distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths) < min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del (filtered_dataset[i])

    return filtered_dataset


def train(args, sess, epoch, image_list, soft_labels_array, index_dequeue_op, enqueue_op, image_paths_placeholder,
          soft_labels_placeholder, learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
          loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file, logits):
    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    index_epoch = sess.run(index_dequeue_op)
    soft_label_epoch = soft_labels_array[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    soft_label_array = soft_label_epoch
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    # persisted_result = sess.graph.get_tensor_by_name("InceptionResnetV1/Block8/Conv2d_1x1/weights:0")
    # tf.stop_gradient(persisted_result)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array,
                          soft_labels_placeholder: soft_label_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True,
                     batch_size_placeholder: args.batch_size}
        if (batch_number % 100 == 0):
            err, _, step, reg_loss, summary_str = sess.run(
                [loss, train_op, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss = sess.run([loss, train_op, global_step, regularization_losses], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
              (epoch, batch_number + 1, args.epoch_size, duration, err, np.sum(reg_loss)))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    # print(logits.eval())
    return step


def evaluate(sess, enqueue_op, image_paths_placeholder, soft_labels_placeholder, phase_train_placeholder,
             batch_size_placeholder,
             embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer,
             nrof_classes, regularization_losses):
    print(batch_size)
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnningforward pass on LFW images')

    # Enqueue one epoch of image paths and labels
    labels_array = np.ndarray(shape=(len(image_paths), nrof_classes))
    image_paths_array = np.expand_dims(np.array(image_paths), 1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, soft_labels_placeholder: labels_array})

    embedding_size = embeddings.get_shape()[1]
    nrof_images = len(actual_issame) * 2
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))

    for _ in range(nrof_batches):
        feed_dict = {phase_train_placeholder: False, batch_size_placeholder: batch_size}
        emb, lab, reg_loss = sess.run([embeddings, labels, regularization_losses], feed_dict=feed_dict)

        lab_array[_] = _
        emb_array[_] = emb

    assert np.array_equal(lab_array, np.arange(
        nrof_images)) == True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    # print(labels)
    # print(lab)
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     labels=lab, logits=err, name='cross_entropy_per_example')
    # cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # err = sess.run(cross_entropy_mean)
    regloss = np.sum(reg_loss)
    accuracy = np.mean(accuracy)
    print('Evaluate: RegLoss %2.3f' % regloss)
    print('Accuracy: %1.3f+-%1.3f' % (accuracy, np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=accuracy)
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    # summary.value.add(tag='accuracy', simple_value=accuracy)
    summary.value.add(tag='lfw/center_loss', simple_value=regloss)
    summary_writer.add_summary(summary, step)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, 'lfw_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, accuracy, val))
    return accuracy


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def get_image_paths_and_labels(path):
    path = os.path.join(os.path.dirname(os.getcwd()), path)
    all_path = os.listdir(path)
    labels = []
    paths = []
    for people_path in all_path:
        for pic in os.listdir(os.path.join(path, people_path)):
            labels.append(int(people_path))
            paths.append(os.path.join(path, people_path, pic))
    image_paths_flat = []
    labels_flat = []
    for i in range(len(paths)):
        image_paths_flat += paths[i]
        labels_flat += [i] * len(paths[i])
    return paths, labels, len(all_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.',
                        default='4_2_512/model-20180402-114759.ckpt-275')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='data_folder')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='inception_resnet_v1')

    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=10)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=100)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=10)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=512)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate',
                        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--fuzzier', type=float,
                        help='fuzzier parameter', default=2)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
                        help='Center loss factor.', default=0.1)
    parser.add_argument('--center_loss_alfa', type=float,
                        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.01)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
                        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
                        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
                        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
                        help='Keep only the classes with this number of examples or more', default=0)

    # Parameters for validation on LFW
    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.', default='../2017/test')
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=1)
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=4)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    tools.email_subject(os.path.basename(__file__))