import pickle
import tensorflow as tf

def data_loader(data_path, batch_size=128, only_video=False, only_eeg=False):
    # 从文件中加载数据
    print("loading:", data_path)
    with open(data_path, 'rb') as f:
        load_data = pickle.load(f)
    train_set = load_data['train']
    valid_set = load_data['valid']
    test_set = load_data['test']

    # 打印数据集信息
    print('train key', train_set.keys())
    print("train_set['label']", train_set['label'].shape)
    print('train_set["EEG"], train_set["video"]', train_set["EEG"].shape, train_set["video"].shape)

    # 使用 TensorFlow 构建数据集对象
    x_train_video = train_set["video"]
    x_train_eeg = train_set["EEG"]
    y_train = train_set["label"]

    x_valid_video = valid_set["video"]
    x_valid_eeg = valid_set["EEG"]
    y_valid = valid_set["label"]

    x_test_video = test_set["video"]
    x_test_eeg = test_set["EEG"]
    y_test = test_set["label"]

    if not only_video and not only_eeg:
        # 构建 TensorFlow 数据集对象
        print("x_train_video, x_train_eeg", x_train_video.shape, x_train_eeg.shape)
        train_dataset = tf.data.Dataset.from_tensor_slices(((x_train_video, x_train_eeg), y_train)).shuffle(10000).batch(batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices(((x_valid_video, x_valid_eeg), y_valid)).shuffle(500).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(((x_test_video, x_test_eeg), y_test)).shuffle(500).batch(batch_size)
    elif only_video:
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_video, y_train)).shuffle(10000).batch(batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid_video, y_valid)).shuffle(500).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test_video, y_test)).shuffle(500).batch(batch_size)
    elif only_eeg:
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_eeg, y_train)).shuffle(10000).batch(batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid_eeg, y_valid)).shuffle(500).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test_eeg, y_test)).shuffle(500).batch(batch_size)

    return train_dataset, valid_dataset, test_dataset