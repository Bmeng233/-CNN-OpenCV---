import Input_module
import CNN_Core
import tensorflow as tf

with tf.Graph().as_default():
    IMG_W = 227
    IMG_H = 227
    BATCH_SIZE = 500
    CAPACITY = 2000
    N_CLASSES = 5
    test_dir = 'E:\Python project\Machine-Learning\data\log'
    test, test_label = Input_module.get_files(test_dir)
    test_batch, test_label_batch = Input_module.get_batch(test,
                                                          test_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)

    test_logit = CNN_Core.cnn_inference(test_batch, BATCH_SIZE, N_CLASSES, keep_prob=1)
    test_acc = CNN_Core.evaluation(test_logit, test_label_batch)
    test_loss = CNN_Core.losses(test_logit, test_label_batch)

    logs_train_dir = 'E:\Python project\Machine-Learning\data\log'
    saver = tf.train.Saver()

    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.all_model_checkpoint_paths:
            for path in ckpt.all_model_checkpoint_paths:
                saver.restore(sess, path)
                global_step = path.split('/')[-1].split('-')[-1]
                print('Loading success, global_step is %s' % global_step)
                accuracy, loss = sess.run([test_acc, test_loss])
                print("测试集正确率是：%.2f%%" % (accuracy * 100))
                print("测试集损失率：%.2f" % loss)
        else:
            print('No checkpoint file found')




