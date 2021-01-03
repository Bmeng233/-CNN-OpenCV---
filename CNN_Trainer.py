import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import CNN_Core
import Input_module


N_CLASSES = 5
IMG_W = 100  # 图像大小
IMG_H = 100
BATCH_SIZE = 32  # 每次训练的张数
CAPACITY = 320
MAX_STEP = 1000
learning_rate = 0.0001  # 学习率
train_dir = 'E:\Python project\Machine-Learning\data\New_test_image\roi'
logs_train_dir = 'E:\Python project\Machine-Learning\data\log'
train, train_label = Input_module.get_files(train_dir)
train_batch, train_label_batch = Input_module.get_batch(train,
                                                        train_label,
                                                        IMG_W,
                                                        IMG_H,
                                                        BATCH_SIZE,
                                                        CAPACITY)

train_logits = CNN_Core.cnn_inference(train_batch, BATCH_SIZE, N_CLASSES, keep_prob=0.5)
train_loss = CNN_Core.losses(train_logits, train_label_batch)
train_op = CNN_Core.training(train_loss, learning_rate)
train__acc = CNN_Core.evaluation(train_logits, train_label_batch)
summary_op = tf.summary.merge_all()

# 显示可视化折线图
step_list = list(range(50))
cnn_list1 = []
cnn_list2 = []
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.yaxis.grid(True)
ax.set_title('cnn_accuracy ', fontsize=14, y=1.02)
ax.set_xlabel('step')
ax.set_ylabel('accuracy')
bx = fig.add_subplot(2, 1, 2)
bx.yaxis.grid(True)
bx.set_title('cnn_loss ', fontsize=14, y=1.02)
bx.set_xlabel('step')
bx.set_ylabel('loss')

# 初始化
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)   #存储log文件
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _op, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
            # 每隔20步打印当前的loss以及acc，同时记录log，写入writer
            if step % 20 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
                cnn_list1.append(tra_acc)
                cnn_list2.append(tra_loss)
            # 每隔40步保存训练好的模型
            if step % 60 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
        ax.plot(step_list, cnn_list1, color="r", label=train)
        bx.plot(step_list, cnn_list2, color="r", label=train)
        plt.tight_layout()
        plt.show()

    except tf.errors.OutOfRangeError:
        print('训练终止-错误')
    finally:
        coord.request_stop()



