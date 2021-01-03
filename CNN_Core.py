import tensorflow as tf


def cnn_inference(images, batch_size, n_classes, keep_prob):

    # 第一层的卷积层convolution1，卷积核使用3X3，共16个
    with tf.variable_scope('convolution1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[11, 11, 3, 96],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[96],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 4, 4, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)        # 偏差
        convolution1 = tf.nn.relu(pre_activation, name=scope.name)  # 激活函数非线性化处理

    # 第一层的池化层pool1和规范化norm1(特征缩放）
    with tf.variable_scope('pooling1_lrn'):
        pool1 = tf.nn.max_pool(convolution1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm1')
        # ksize为池化窗口的大小，步长一般比卷积核多移动一位

    # 第二层的卷积层convolution2
    with tf.variable_scope('convolution2'):
        weights = tf.get_variable('weights',
                                  shape=[5, 5, 96, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        convolution2 = tf.nn.relu(pre_activation, name='convolution2')

    # 第二层的池化层pool2和规范化norm2
    with tf.variable_scope('pooling2_lrn'):
        norm2 = tf.nn.lrn(convolution2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID',name='pooling2')  # 先规范化再池化

    # 第三层的卷积层convolution3
    with tf.variable_scope('convolution3'):
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 256, 384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        convolution3 = tf.nn.relu(pre_activation, name='convolution3')

    # 第四层的卷积层convolution4
    with tf.variable_scope('convolution4'):
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 384, 384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(convolution3, weights, strides=[1, 1, 1, 1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        convolution4 = tf.nn.relu(pre_activation, name='convolution4')

    # 第五层的卷积层convolution5
    with tf.variable_scope('convolution5'):
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 384, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(convolution4, weights, strides=[1, 1, 1, 1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        convolution5 = tf.nn.relu(pre_activation, name='convolution5')

    # 池化
    with tf.variable_scope('pooling'):
        pooling = tf.nn.max_pool(convolution5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                 padding='VALID', name='pooling1')

    # 第三层：全连接层local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pooling, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 1024],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[1024],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)  # 矩阵相乘加上bias
        local3 = tf.nn.dropout(local3, keep_prob)  # 设置神经元被选中的概率

    # 第四层：全连接层local4
    with tf.variable_scope('local4'):
        weights = tf.get_variable('weights',
                                  shape=[1024, 1024],  # 再连接1024个神经元
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[1024],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
        local4 = tf.nn.dropout(local4, keep_prob)

    # 第五层：输出层softmax_linear
    with tf.variable_scope('softmax_linear'):
        weights = tf.get_variable('weights',
                                  shape=[1024, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
        softmax_linear = tf.nn.dropout(softmax_linear, keep_prob)
    return softmax_linear


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        # 提高计算速度
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='loss_per_eg')
        loss = tf.reduce_mean(cross_entropy, name='loss')  # 求样本平均loss
    return loss


def training(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op


def evaluation(logits, labels):

    with tf.variable_scope('accuracy'):
        prediction = tf.nn.softmax(logits)
        correct = tf.nn.in_top_k(prediction, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
    return accuracy
