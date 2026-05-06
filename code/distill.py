import tensorflow as tf
from tensorflow.keras import Model

class Distilling(Model):
    def __init__(self, student_model=None, teacher_model=None, train_mode='teacher-TA_1', **kwargs):
        super(Distilling, self).__init__(**kwargs)
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.train_mode = train_mode

        self.T = 0.
        self.alpha = 0.

        self.clf_loss = None
        self.kd_loss = None
        self.clf_loss_tracker = tf.keras.metrics.Mean(name='clf_loss')
        self.kd_loss_tracker = tf.keras.metrics.Mean(name='kd_loss')
        self.sum_loss_tracker = tf.keras.metrics.Mean(name='loss')

    def compile(self, clf_loss=None, kd_loss=None, T=0., alpha=0.,  **kwargs):
        super(Distilling, self).compile(**kwargs)
        self.clf_loss = clf_loss
        self.kd_loss = kd_loss
        self.T = T
        '''
            温度系数 T 在知识蒸馏中非常重要，通常用来控制 softmax 的平滑程度。
            温度越高，softmax 的输出分布越平滑，
            反之，温度越低，softmax 输出会更接近于 one-hot 分布。
            较高的温度帮助学生模型从教师模型中学习更丰富的信息，尤其是在不同类的相对概率之间的关系上。
        '''
        self.alpha = alpha

    @property
    def metrics(self):
        metrics = [self.sum_loss_tracker, self.clf_loss_tracker, self.kd_loss_tracker]

        if self.compiled_metrics is not None:
            metrics += self.compiled_metrics.metrics

        return metrics

    def train_step(self, data):
        x, y = data
        video_data, eeg_data = x

        with tf.GradientTape() as tape:
            if self.train_mode.endswith('student'):
                student_cls, student_logits, student_att = self.student_model(video_data)
            else:
                student_cls, student_logits = self.student_model(x)
            if self.train_mode.startswith('teacher'):
                teacher_cls, _, _, teacher_logits, teacher_att = self.teacher_model(x, training=False)
            else:
                teacher_cls, teacher_logits = self.teacher_model(x, training=False)

            clf_loss_value = self.clf_loss(y, student_cls)
            kd_loss_value = self.kd_loss(tf.math.softmax(student_logits/self.T), tf.math.softmax(teacher_logits/self.T))
            sum_loss_value = self.alpha * clf_loss_value + (1-self.alpha) * kd_loss_value

        self.optimizer.minimize(sum_loss_value, self.student_model.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, student_cls)

        self.sum_loss_tracker.update_state(sum_loss_value)
        self.clf_loss_tracker.update_state(clf_loss_value)
        self.kd_loss_tracker.update_state(kd_loss_value)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        video_data, eeg_data = x

        if self.train_mode.endswith('student'):
            student_cls, student_logits = self.student_model(video_data, training=False)
        else:
            student_cls, student_logits = self.student_model(x, training=False)
        if self.train_mode.startswith('teacher'):
            teacher_cls, _, _, teacher_logits, _ = self.teacher_model(x, training=False)
        else:
            teacher_cls, teacher_logits = self.teacher_model(x, training=False)

        clf_loss_value = self.clf_loss(y, student_cls)
        kd_loss_value = self.kd_loss(tf.math.softmax(student_logits/self.T), tf.math.softmax(teacher_logits/self.T))
        sum_loss_value = self.alpha * clf_loss_value + (1-self.alpha) * kd_loss_value

        self.compiled_metrics.update_state(y, student_cls)

        self.sum_loss_tracker.update_state(sum_loss_value)
        self.clf_loss_tracker.update_state(clf_loss_value)
        self.kd_loss_tracker.update_state(kd_loss_value)

        return {m.name: m.result() for m in self.metrics}

class SaveBestStudentModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, monitor='val_acc', mode='max'):
        super(SaveBestStudentModelCallback, self).__init__()
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.best = -float('inf') if mode == 'max' else float('inf')  # 初始化最佳值
        self.monitor_op = tf.math.greater if mode == 'max' else tf.math.less  # 比较操作符

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        # 如果当前的 val_acc 比之前最佳的要好，保存学生模型
        if self.monitor_op(current, self.best):
            self.best = current
            print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {current:.4f}, saving student model to {self.save_path}")
            self.model.student_model.save(self.save_path)