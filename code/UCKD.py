import tensorflow as tf
from tensorflow.keras import Model


class UCKD(Model):
    """
    UCKD: Uncertainty-aware Cross-modal Knowledge Distillation

    核心思想：
    1. 去掉随机 drop
    2. 只使用学生模型所用模态的预测不确定性来调节 KD 强度
    3. 多教师权重由教师群体一致性决定
    4. 保持原先训练接口基本不变
    """

    def __init__(
        self,
        student_model=None,
        teacher_models=None,
        train_mode='teacher-TA_1',
        T=2,
        alpha=1,
        droprate=0.5,      # 为兼容旧接口保留，但不再使用
        agreement_beta=1.0,
        eps=1e-8,
        **kwargs
    ):
        super(UCKD, self).__init__(**kwargs)
        self.student_model = student_model
        self.teacher_models = teacher_models
        self.train_mode = train_mode

        self.T = T
        self.alpha = alpha
        self.droprate = droprate   # 仅保留接口，不参与计算
        self.agreement_beta = agreement_beta
        self.eps = eps

        self.clf_loss = None
        self.kd_loss = None  # 为兼容 compile 接口保留

        self.clf_loss_tracker = tf.keras.metrics.Mean(name='clf_loss')
        self.kd_loss_tracker = tf.keras.metrics.Mean(name='kd_loss')
        self.sum_loss_tracker = tf.keras.metrics.Mean(name='loss')

    def compile(self, clf_loss=None, kd_loss=None, **kwargs):
        super(UCKD, self).compile(**kwargs)
        self.clf_loss = clf_loss
        self.kd_loss = kd_loss  # 保留接口；实际 KD 用内部实现

    @property
    def metrics(self):
        metrics = [self.sum_loss_tracker, self.clf_loss_tracker, self.kd_loss_tracker]
        if self.compiled_metrics is not None:
            metrics += self.compiled_metrics.metrics
        return metrics

    # =========================
    # Output parsing utilities
    # =========================
    def _parse_model_outputs(self, outputs, role='student', idx=None):
        """
        统一解析 student / teacher 的输出格式。

        支持：
        1) (cls, logits)
        2) (cls, ..., ..., logits, ...)
        """
        if not isinstance(outputs, (tuple, list)):
            raise ValueError(
                f"{role}_model output must be tuple/list, but got type: {type(outputs)}"
            )

        if len(outputs) == 2:
            cls, logits = outputs
        elif len(outputs) >= 4:
            cls = outputs[0]
            logits = outputs[3]
        else:
            if role == 'teacher':
                raise ValueError(
                    f"Unexpected teacher model output format at index {idx}: "
                    f"got {len(outputs)} outputs."
                )
            else:
                raise ValueError(
                    f"Unexpected student model output format: got {len(outputs)} outputs."
                )

        return cls, logits

    # =========================
    # Forward functions
    # =========================
    def _forward_student(self, x):
        """
        与原 DGKD 保持一致地区分 student 阶段和 TA 阶段：
        - 最终 student 阶段：student_model = VideoModel，只输入 video_data
        - 其他阶段：student_model = TA model，输入完整 x
        """
        video_data, eeg_data = x

        if self.train_mode.endswith('student'):
            outputs = self.student_model(video_data, training=True)

            if not isinstance(outputs, (tuple, list)):
                raise ValueError(
                    f"Video student model output must be tuple/list, but got {type(outputs)}"
                )

            # 你的 VideoModel: return cls, logits, att_feature
            if len(outputs) == 3:
                student_cls = outputs[0]
                student_logits = outputs[1]
            elif len(outputs) == 2:
                student_cls = outputs[0]
                student_logits = outputs[1]
            else:
                raise ValueError(
                    f"Unexpected VideoModel output format: got {len(outputs)} outputs."
                )

        else:
            outputs = self.student_model(x, training=True)

            if not isinstance(outputs, (tuple, list)):
                raise ValueError(
                    f"TA student model output must be tuple/list, but got {type(outputs)}"
                )

            # TA阶段尽量兼容原来的多种写法
            if len(outputs) == 2:
                student_cls = outputs[0]
                student_logits = outputs[1]
            elif len(outputs) == 3:
                # 若某个 TA 返回 (cls, logits, feature)
                student_cls = outputs[0]
                student_logits = outputs[1]
            elif len(outputs) >= 4:
                # 兼容 (cls, ..., ..., logits, ...)
                student_cls = outputs[0]
                student_logits = outputs[3]
            else:
                raise ValueError(
                    f"Unexpected TA student output format: got {len(outputs)} outputs."
                )

        return student_cls, student_logits

    def _forward_student_test(self, x):
        video_data, eeg_data = x

        if self.train_mode.endswith('student'):
            outputs = self.student_model(video_data, training=False)

            if not isinstance(outputs, (tuple, list)):
                raise ValueError(
                    f"Video student model output must be tuple/list, but got {type(outputs)}"
                )

            # 你的 VideoModel: return cls, logits, att_feature
            if len(outputs) == 3:
                student_cls = outputs[0]
                student_logits = outputs[1]
            elif len(outputs) == 2:
                student_cls = outputs[0]
                student_logits = outputs[1]
            else:
                raise ValueError(
                    f"Unexpected VideoModel output format: got {len(outputs)} outputs."
                )

        else:
            outputs = self.student_model(x, training=False)

            if not isinstance(outputs, (tuple, list)):
                raise ValueError(
                    f"TA student model output must be tuple/list, but got {type(outputs)}"
                )

            if len(outputs) == 2:
                student_cls = outputs[0]
                student_logits = outputs[1]
            elif len(outputs) == 3:
                student_cls = outputs[0]
                student_logits = outputs[1]
            elif len(outputs) >= 4:
                student_cls = outputs[0]
                student_logits = outputs[3]
            else:
                raise ValueError(
                    f"Unexpected TA student output format: got {len(outputs)} outputs."
                )

        return student_cls, student_logits

    def _forward_teacher(self, teacher, x, idx):
        """
        统一解析 teacher 输出。
        """
        outputs = teacher(x, training=False)
        return self._parse_model_outputs(outputs, role='teacher', idx=idx)

    # =========================
    # Uncertainty & KD utilities
    # =========================
    def _normalized_entropy(self, prob):
        """
        prob: [B, C]
        return: [B]
        """
        num_classes = tf.cast(tf.shape(prob)[-1], tf.float32)
        entropy = -tf.reduce_sum(prob * tf.math.log(prob + self.eps), axis=-1)
        max_entropy = tf.math.log(num_classes + self.eps)
        return entropy / (max_entropy + self.eps)

    def _teacher_agreement_weights(self, teacher_probs):
        """
        teacher_probs: [N, B, C]

        基于教师群体一致性计算样本级 teacher 权重：
        1. mean teacher distribution
        2. 每个 teacher 到 mean distribution 的 MSE
        3. agreement score = exp(-beta * mse)
        4. 在 teacher 维度归一化

        return:
            teacher_weights: [N, B]
        """
        mean_teacher_prob = tf.reduce_mean(teacher_probs, axis=0)  # [B, C]

        # [N, B]
        mse_to_mean = tf.reduce_mean(
            tf.square(teacher_probs - mean_teacher_prob[None, :, :]),
            axis=-1
        )

        agreement_scores = tf.exp(-self.agreement_beta * mse_to_mean)  # [N, B]

        teacher_weights = agreement_scores / (
            tf.reduce_sum(agreement_scores, axis=0, keepdims=True) + self.eps
        )
        return teacher_weights

    def _per_sample_kl(self, teacher_prob, student_prob):
        """
        KL(teacher || student), return: [B]
        """
        return tf.reduce_sum(
            teacher_prob * (
                tf.math.log(teacher_prob + self.eps) -
                tf.math.log(student_prob + self.eps)
            ),
            axis=-1
        )

    def _compute_uncertainty_aware_kd(self, x, student_logits):
        """
        不确定性感知的多教师 KD：

        - 学生侧：只使用学生模型所用数据的不确定性
        - 教师侧：只使用教师群体一致性进行加权
        """
        # 学生软分布 [B, C]
        student_prob = tf.nn.softmax(student_logits / self.T, axis=-1)

        # 学生不确定性 [B]
        # 熵越大，不确定性越高，KD 强度越大
        student_uncertainty = self._normalized_entropy(student_prob)

        teacher_probs = []
        for idx, teacher in enumerate(self.teacher_models):
            _, teacher_logits = self._forward_teacher(teacher, x, idx)
            teacher_prob = tf.nn.softmax(teacher_logits / self.T, axis=-1)
            teacher_probs.append(teacher_prob)

        # [N, B, C]
        teacher_probs = tf.stack(teacher_probs, axis=0)

        # [N, B]
        teacher_weights = self._teacher_agreement_weights(teacher_probs)

        # [N, B]
        kd_per_teacher = []
        for i in range(len(self.teacher_models)):
            kd_i = self._per_sample_kl(teacher_probs[i], student_prob)
            kd_per_teacher.append(kd_i)

        kd_per_teacher = tf.stack(kd_per_teacher, axis=0)

        # [B]
        aggregated_teacher_kd = tf.reduce_sum(teacher_weights * kd_per_teacher, axis=0)

        # [B]
        weighted_kd = student_uncertainty * aggregated_teacher_kd

        # scalar
        total_kd_loss = tf.reduce_mean(weighted_kd)

        # 经典 KD 温度补偿
        total_kd_loss = (self.T ** 2) * total_kd_loss

        return total_kd_loss

    # =========================
    # Train / test step
    # =========================
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            # 这里已经在 _forward_student 里区分了最终student阶段和TA阶段
            student_cls, student_logits = self._forward_student(x)

            # 分类损失
            clf_loss_value = self.clf_loss(y, student_cls)

            # 不确定性感知KD损失
            kd_loss_value = self._compute_uncertainty_aware_kd(x, student_logits)

            # 总损失
            sum_loss_value = self.alpha * clf_loss_value + (1.0 - self.alpha) * kd_loss_value

        self.optimizer.minimize(
            sum_loss_value,
            self.student_model.trainable_variables,
            tape=tape
        )

        self.compiled_metrics.update_state(y, student_cls)

        self.sum_loss_tracker.update_state(sum_loss_value)
        self.clf_loss_tracker.update_state(clf_loss_value)
        self.kd_loss_tracker.update_state(kd_loss_value)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        student_cls, student_logits = self._forward_student_test(x)

        clf_loss_value = self.clf_loss(y, student_cls)
        kd_loss_value = self._compute_uncertainty_aware_kd(x, student_logits)

        sum_loss_value = self.alpha * clf_loss_value + (1.0 - self.alpha) * kd_loss_value

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
        self.best = -float('inf') if mode == 'max' else float('inf')
        self.monitor_op = tf.math.greater if mode == 'max' else tf.math.less

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.monitor_op(current, self.best):
            self.best = current
            print(
                f"\nEpoch {epoch + 1}: {self.monitor} improved to {current:.4f}, "
                f"saving student model to {self.save_path}"
            )
            self.model.student_model.save(self.save_path)