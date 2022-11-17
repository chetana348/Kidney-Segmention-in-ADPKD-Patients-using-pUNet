'''

'''
import os
from nnunet2d.layer import (conv2d, deconv2d, normalizationlayer, crop_and_concat, resnet_Add,
                          weight_xavier_init, bias_variable, save_images)
import tensorflow as tf
import numpy as np
import cv2


def conv_bn_relu_drop(x, kernalshape, phase, drop_conv, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernalshape, n_inputs=kernalshape[0] * kernalshape[1] * kernalshape[2],
                               n_outputs=kernalshape[-1], activefunction='relu', variable_name=str(scope) + 'W')
        B = bias_variable([kernalshape[-1]], variable_name=str(scope) + 'B')
        conv = conv2d(x, W) + B
        conv = normalizationlayer(conv, phase, height=height, width=width, norm_type='group', scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop_conv)
        return conv


def down_sampling(x, kernalshape, phase, drop_conv, height=None, width=None, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernalshape, n_inputs=kernalshape[0] * kernalshape[1] * kernalshape[2],
                               n_outputs=kernalshape[-1], activefunction='relu', variable_name=str(scope) + 'W')
        B = bias_variable([kernalshape[-1]], variable_name=str(scope) + 'B')
        conv = conv2d(x, W, 2) + B
        conv = normalizationlayer(conv, phase, height=height, width=width, norm_type='group', scope=scope)
        conv = tf.nn.dropout(tf.nn.relu(conv), drop_conv)
        return conv


def deconv_relu_drop(x, kernalshape, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernalshape, n_inputs=kernalshape[0] * kernalshape[1] * kernalshape[-1],
                               n_outputs=kernalshape[-2], activefunction='relu', variable_name=str(scope) + 'W')
        B = bias_variable([kernalshape[-2]], variable_name=str(scope) + 'B')
        dconv = tf.nn.relu(deconv2d(x, W) + B)
        return dconv


def conv_sigmod(x, kernalshape, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernalshape, n_inputs=kernalshape[0] * kernalshape[1] * kernalshape[2],
                               n_outputs=kernalshape[-1], activefunction='sigomd', variable_name=str(scope) + 'W')
        B = bias_variable([kernalshape[-1]], variable_name=str(scope) + 'B')
        conv = conv2d(x, W) + B
        conv = tf.nn.sigmoid(conv)
        return conv


def conv_relu(x, kernalshape, scope=None):
    with tf.name_scope(scope):
        W = weight_xavier_init(shape=kernalshape, n_inputs=kernalshape[0] * kernalshape[1] * kernalshape[2],
                               n_outputs=kernalshape[-1], activefunction='relu', variable_name=str(scope) + 'W')
        B = bias_variable([kernalshape[-1]], variable_name=str(scope) + 'B')
        conv = conv2d(x, W) + B
        conv = tf.nn.relu(conv)
        return conv


def _create_conv_net(X, image_width, image_height, image_channel, phase, drop_conv, n_class=1):
    inputX = tf.reshape(X, [-1, image_width, image_height, image_channel])  # shape=(?, 32, 32, 1)
    # pUNet model
    # layer1->convolution
    layer0 = conv_relu(x=inputX, kernalshape=(3, 3, image_channel, 16), scope='layer0')
    layer1 = conv_bn_relu_drop(x=layer0, kernalshape=(3, 3, 16, 16), phase=phase, drop_conv=drop_conv,
                               scope='layer1')
    layer1 = resnet_Add(x1=layer0, x2=layer1)
    # down sampling1
    down1 = down_sampling(x=layer1, kernalshape=(3, 3, 16, 32), phase=phase, drop_conv=drop_conv, scope='down1')
    # layer2->convolution
    layer2 = conv_bn_relu_drop(x=down1, kernalshape=(3, 3, 32, 32), phase=phase, drop_conv=drop_conv,
                               scope='layer2_1')
    layer2 = conv_bn_relu_drop(x=layer2, kernalshape=(3, 3, 32, 32), phase=phase, drop_conv=drop_conv,
                               scope='layer2_2')
    layer2 = resnet_Add(x1=down1, x2=layer2)
    # down sampling2
    down2 = down_sampling(x=layer2, kernalshape=(3, 3, 32, 64), phase=phase, drop_conv=drop_conv, scope='down2')
    # layer3->convolution
    layer3 = conv_bn_relu_drop(x=down2, kernalshape=(3, 3, 64, 64), phase=phase, drop_conv=drop_conv,
                               scope='layer3_1')
    layer3 = conv_bn_relu_drop(x=layer3, kernalshape=(3, 3, 64, 64), phase=phase, drop_conv=drop_conv,
                               scope='layer3_2')
    layer3 = conv_bn_relu_drop(x=layer3, kernalshape=(3, 3, 64, 64), phase=phase, drop_conv=drop_conv,
                               scope='layer3_3')
    layer3 = resnet_Add(x1=down2, x2=layer3)
    # down sampling3
    down3 = down_sampling(x=layer3, kernalshape=(3, 3, 64, 128), phase=phase, drop_conv=drop_conv, scope='down3')
    # layer4->convolution
    layer4 = conv_bn_relu_drop(x=down3, kernalshape=(3, 3, 128, 128), phase=phase, drop_conv=drop_conv,
                               scope='layer4_1')
    layer4 = conv_bn_relu_drop(x=layer4, kernalshape=(3, 3, 128, 128), phase=phase, drop_conv=drop_conv,
                               scope='layer4_2')
    layer4 = conv_bn_relu_drop(x=layer4, kernalshape=(3, 3, 128, 128), phase=phase, drop_conv=drop_conv,
                               scope='layer4_3')
    layer4 = resnet_Add(x1=down3, x2=layer4)
    # down sampling4
    down4 = down_sampling(x=layer4, kernalshape=(3, 3, 128, 256), phase=phase, drop_conv=drop_conv, scope='down4')
    # layer5->convolution
    layer5 = conv_bn_relu_drop(x=down4, kernalshape=(3, 3, 256, 256), phase=phase, drop_conv=drop_conv,
                               scope='layer5_1')
    layer5 = conv_bn_relu_drop(x=layer5, kernalshape=(3, 3, 256, 256), phase=phase, drop_conv=drop_conv,
                               scope='layer5_2')
    layer5 = conv_bn_relu_drop(x=layer5, kernalshape=(3, 3, 256, 256), phase=phase, drop_conv=drop_conv,
                               scope='layer5_3')
    layer5 = resnet_Add(x1=down4, x2=layer5)
    # down sampling5
    down5 = down_sampling(x=layer5, kernalshape=(3, 3, 256, 512), phase=phase, drop_conv=drop_conv, scope='down5')
    # layer6->convolution
    layer6 = conv_bn_relu_drop(x=down5, kernalshape=(3, 3, 512, 512), phase=phase, drop_conv=drop_conv,
                               scope='layer6_1')
    layer6 = conv_bn_relu_drop(x=layer6, kernalshape=(3, 3, 512, 512), phase=phase, drop_conv=drop_conv,
                               scope='layer6_2')
    layer6 = resnet_Add(x1=down5, x2=layer6)

    # layer7->deconvolution
    deconv1 = deconv_relu_drop(x=layer6, kernalshape=(3, 3, 256, 512), scope='deconv1')
    # layer8->convolution
    layer7 = crop_and_concat(layer5, deconv1)
    _, H, W, _ = layer5.get_shape().as_list()
    layer7 = conv_bn_relu_drop(x=layer7, kernalshape=(3, 3, 512, 256), height=H, width=W, phase=phase,
                               drop_conv=drop_conv, scope='layer7_1')
    layer7 = conv_bn_relu_drop(x=layer7, kernalshape=(3, 3, 256, 256), height=H, width=W, phase=phase,
                               drop_conv=drop_conv, scope='layer7_2')
    layer7 = resnet_Add(x1=deconv1, x2=layer7)

    # layer9->deconvolution
    deconv2 = deconv_relu_drop(x=layer7, kernalshape=(3, 3, 128, 256), scope='deconv2')
    # layer8->convolution
    layer8 = crop_and_concat(layer4, deconv2)
    _, H, W, _ = layer4.get_shape().as_list()
    layer8 = conv_bn_relu_drop(x=layer8, kernalshape=(3, 3, 256, 128), height=H, width=W, phase=phase,
                               drop_conv=drop_conv, scope='layer8_1')
    layer8 = conv_bn_relu_drop(x=layer8, kernalshape=(3, 3, 128, 128), height=H, width=W, phase=phase,
                               drop_conv=drop_conv, scope='layer8_2')
    layer8 = resnet_Add(x1=deconv2, x2=layer8)

    # layer9->deconvolution
    deconv3 = deconv_relu_drop(x=layer8, kernalshape=(3, 3, 64, 128), scope='deconv3')
    # layer8->convolution
    layer9 = crop_and_concat(layer3, deconv3)
    _, H, W, _ = layer3.get_shape().as_list()
    layer9 = conv_bn_relu_drop(x=layer9, kernalshape=(3, 3, 128, 64), height=H, width=W, phase=phase,
                               drop_conv=drop_conv, scope='layer9_1')
    layer9 = conv_bn_relu_drop(x=layer9, kernalshape=(3, 3, 64, 64), height=H, width=W, phase=phase,
                               drop_conv=drop_conv, scope='layer9_2')
    layer9 = resnet_Add(x1=deconv3, x2=layer9)

    # layer9->deconvolution
    deconv4 = deconv_relu_drop(x=layer9, kernalshape=(3, 3, 32, 64), scope='deconv4')
    # layer8->convolution
    layer10 = crop_and_concat(layer2, deconv4)
    _, H, W, _ = layer2.get_shape().as_list()
    layer10 = conv_bn_relu_drop(x=layer10, kernalshape=(3, 3, 64, 32), height=H, width=W, phase=phase,
                                drop_conv=drop_conv, scope='layer10_1')
    layer10 = conv_bn_relu_drop(x=layer10, kernalshape=(3, 3, 32, 32), height=H, width=W, phase=phase,
                                drop_conv=drop_conv, scope='layer10_2')
    layer10 = resnet_Add(x1=deconv4, x2=layer10)

    # layer9->deconvolution
    deconv5 = deconv_relu_drop(x=layer10, kernalshape=(3, 3, 16, 32), scope='deconv5')
    # layer8->convolution
    layer11 = crop_and_concat(layer1, deconv5)
    _, H, W, _ = layer1.get_shape().as_list()
    layer11 = conv_bn_relu_drop(x=layer11, kernalshape=(3, 3, 32, 16), height=H, width=W, phase=phase,
                                drop_conv=drop_conv, scope='layer11_1')
    layer11 = conv_bn_relu_drop(x=layer11, kernalshape=(3, 3, 16, 16), height=H, width=W, phase=phase,
                                drop_conv=drop_conv, scope='layer11_2')
    layer11 = resnet_Add(x1=deconv5, x2=layer11)
    # layer14->output
    output_map = conv_sigmod(x=layer11, kernalshape=(1, 1, 16, n_class), scope='output')
    return output_map


def _next_batch(train_images, train_labels, batch_size, index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size

    num_examples = train_images.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], index_in_epoch


class BLpUnet2dModule(object):
    """
    A pUNet2d implementation

    :param image_height: number of height in the input image
    :param image_width: number of width in the input image
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param costname: name of the cost function.Default is "cross_entropy"
    """

    def __init__(self, image_height, image_width, channels=1, inference=False, model_path=None,
                 costname=("dice coefficient",)):
        self.image_with = image_width
        self.image_height = image_height
        self.channels = channels

        self.X = tf.placeholder("float", shape=[None, image_height, image_width, channels], name="Input")
        self.Y_gt = tf.placeholder("float", shape=[None, image_height, image_width, channels], name="Output_GT")
        self.lr = tf.placeholder('float', name="Learning_rate")
        self.phase = tf.placeholder(tf.bool, name="Phase")
        self.drop_conv = tf.placeholder('float', name="DropOut")

        self.Y_pred = _create_conv_net(self.X,
                                       image_width,
                                       image_height,
                                       channels,
                                       self.phase,
                                       self.drop_conv)

        self.cost = self.__get_cost(costname[0], self.Y_gt, self.Y_pred)
        self.accuracy = -self.__get_cost(costname[0], self.Y_gt, self.Y_pred)
        if inference:
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            self.sess = tf.InteractiveSession()
            self.sess.run(init)
            saver.restore(self.sess, model_path)

    def __get_cost(self, cost_name, Y_pred, Y_gt):
        H, W, C = self.Y_gt.get_shape().as_list()[1:]
        if cost_name == "dice coefficient":
            smooth = 1e-5
            pred_flat = tf.reshape(Y_pred, [-1, H * W * C])
            true_flat = tf.reshape(Y_gt, [-1, H * W * C])
            intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
            denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
            loss = -tf.reduce_mean(intersection / denominator)
            return loss
        if cost_name == "twersky_loss":
            smooth = 1e-5
            alpha = 0.5
            beta = 0.5
            ones = tf.ones(tf.shape(Y_gt))
            p0 = Y_pred
            p1 = ones - Y_pred
            g0 = Y_gt
            g1 = ones - Y_gt
            num = tf.reduce_sum(p0 * g0, axis=(0, 1, 2))
            den = num + alpha * tf.reduce_sum(p0 * g1, axis=(0, 1, 2)) + \
                  beta * tf.reduce_sum(p1 * g0, axis=(0, 1, 2)) + smooth
            T = tf.reduce_sum(num / den)
            Ncl = tf.cast(tf.shape(Y_gt)[-1], 'float32')
            loss = Ncl - T
            return loss

    def train(self, train_images, train_lanbels, model_path, logs_path, learning_rate,
              dropout_conv=0.8, train_epochs=1000, batch_size=1):
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if not os.path.exists(logs_path + "model\\"):
            os.makedirs(logs_path + "model\\")
        model_path = logs_path + "model\\" + model_path
        # update the moving average of batch norm before finish the training step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # ensures that we execute the update_ops before performing the train_step
            train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        merged_summary_op = tf.summary.merge_all()
        sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        sess.run(init)

        DISPLAY_STEP = 1
        index_in_epoch = 0

        train_epochs = train_images.shape[0] * train_epochs
        for i in range(train_epochs):
            # get new batch
            batch_xs_path, batch_ys_path, index_in_epoch = _next_batch(train_images, train_lanbels, batch_size,
                                                                       index_in_epoch)
            batch_xs = np.empty((len(batch_xs_path), self.image_height, self.image_with, self.channels))
            batch_ys = np.empty((len(batch_ys_path), self.image_height, self.image_with, self.channels))

            for num in range(len(batch_xs_path)):
                image = cv2.imread(batch_xs_path[num], cv2.IMREAD_GRAYSCALE)
                label = cv2.imread(batch_ys_path[num], cv2.IMREAD_GRAYSCALE)
                batch_xs[num, :, :, :] = np.reshape(image, (self.image_height, self.image_with, self.channels))
                batch_ys[num, :, :, :] = np.reshape(label, (self.image_height, self.image_with, self.channels))
            # Extracting images and labels from given data
            batch_xs = batch_xs.astype(np.float)
            batch_ys = batch_ys.astype(np.float)
            # Normalize from [0:255] => [0.0:1.0]
            batch_xs = np.multiply(batch_xs, 1.0 / 255.0)
            batch_ys = np.multiply(batch_ys, 1.0 / 255.0)
            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
            if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
                train_loss, train_accuracy = sess.run([self.cost, self.accuracy], feed_dict={self.X: batch_xs,
                                                                                             self.Y_gt: batch_ys,
                                                                                             self.lr: learning_rate,
                                                                                             self.phase: 1,
                                                                                             self.drop_conv: dropout_conv})
                pred = sess.run(self.Y_pred, feed_dict={self.X: batch_xs,
                                                        self.Y_gt: batch_ys,
                                                        self.phase: 1,
                                                        self.drop_conv: 1})
                result = np.reshape(pred, (6, 512, 512))
                result = result.astype(np.float32)
                save_images(result, [3, 2], path=logs_path + 'result_%d_epoch.png' % (i))

                gt = np.reshape(batch_ys, (6, 512, 512))
                gt = gt.astype(np.float32)
                save_images(gt, [3, 2], path=logs_path + 'gt_%d_epoch.png' % (i))
                print('epochs %d training_loss ,Training_accuracy => %.5f,%.5f ' % (i, train_loss, train_accuracy))
                save_path = saver.save(sess, model_path, global_step=i)
                print("Model saved in file:", save_path)
                if i % (DISPLAY_STEP * 10) == 0 and i:
                    DISPLAY_STEP *= 10

                    # train on batch
            _, summary = sess.run([train_op, merged_summary_op], feed_dict={self.X: batch_xs,
                                                                            self.Y_gt: batch_ys,
                                                                            self.lr: learning_rate,
                                                                            self.phase: 1,
                                                                            self.drop_conv: dropout_conv})
            summary_writer.add_summary(summary, i)
        summary_writer.close()

        save_path = saver.save(sess, model_path)
        print("Model saved in file:", save_path)

    def prediction(self, test_images):
        test_images = test_images.astype(np.float)
        # convert from [0:255] => [0.0:1.0]
        test_images = np.multiply(test_images, 1.0 / 255.0)
        test_images = np.reshape(test_images, (1, test_images.shape[0], test_images.shape[1], 1))
        pred = self.sess.run(self.Y_pred, feed_dict={self.X: test_images,
                                                     self.Y_gt: test_images,
                                                     self.phase: 1,
                                                     self.drop_conv: 1})
        result = np.reshape(pred, (test_images.shape[1], test_images.shape[2]))
        result = result.astype(np.float32) * 255.
        result = np.clip(result, 0, 255).astype('uint8')
        return result

    def getaccu(self, test_images, test_masks):
        test_images = test_images.astype(np.float)
        # convert from [0:255] => [0.0:1.0]
        test_images = np.multiply(test_images, 1.0 / 255.0)
        test_images = np.reshape(test_images, (1, test_images.shape[0], test_images.shape[1], 1))

        test_masks = test_masks.astype(np.float)
        # convert from [0:255] => [0.0:1.0]
        test_masks = np.multiply(test_masks, 1.0 / 255.0)
        test_masks = np.reshape(test_masks, (1, test_masks.shape[0], test_masks.shape[1], 1))
        pred = self.sess.run(self.accuracy, feed_dict={self.X: test_images,
                                                       self.Y_gt: test_masks,
                                                       self.phase: 1,
                                                       self.drop_conv: 1})
        return pred