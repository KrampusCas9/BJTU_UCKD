import re
import pandas as pd

# 读取txt文件内容
with open('experiment_results.txt', 'r', encoding='utf-8') as file:
    experiment_text = file.read()

# 定义正则表达式模式以提取相关数据，包括科学计数法的匹配
experiment_pattern = r"Experiment with args:\nPMR_loss: (True|False)\nBatch Size: (\d+)\nEpochs: (\d+)\nLearning Rate: ([\d\.eE-]+)\nData Type: (A|V)\nDistance Type: (\w+)\nAlpha: ([\d\.]+)"
best_epoch_pattern = r"Best Epoch: (\d+)\n\s+Train Loss: ([\d\.]+), Train Accuracy: ([\d\.]+)\n\s+Val Loss: ([\d\.]+), Val Accuracy: ([\d\.]+)"
test_results_pattern = r"Test Results:\n\s+Test Loss: ([\d\.]+), Test Accuracy: ([\d\.]+)"

# 查找所有实验块
experiments = re.findall(experiment_pattern, experiment_text)
best_epochs = re.findall(best_epoch_pattern, experiment_text)
test_results = re.findall(test_results_pattern, experiment_text)

# 将结果合并为结构化数据
results = []
for exp, epoch, test in zip(experiments, best_epochs, test_results):
    pmr_loss, batch_size, epochs, lr, data_type, distance_type, alpha = exp
    best_epoch, train_loss, train_acc, val_loss, val_acc = epoch
    test_loss, test_acc = test
    results.append({
        "PMR_loss": pmr_loss,
        "Batch Size": batch_size,
        "Epochs": epochs,
        "Learning Rate": lr,
        "Data Type": data_type,
        "Distance Type": distance_type,
        "Alpha": alpha,
        "Best Epoch": best_epoch,
        "Train Loss": train_loss,
        "Train Accuracy": train_acc,
        "Val Loss": val_loss,
        "Val Accuracy": val_acc,
        "Test Loss": test_loss,
        "Test Accuracy": test_acc
    })

# 将结果转换为DataFrame
df = pd.DataFrame(results)

# 保存为CSV文件
df.to_csv('experiment_results.csv', index=False)

print("实验结果已成功保存到CSV文件：experiment_results.csv")
