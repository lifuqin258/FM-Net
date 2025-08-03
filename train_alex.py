import torch.optim as optim
from main_alex import *
from data_loader import *
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

train_acc_list = []
test_acc_list = []

start_time = time.time()

model = FM()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('G:\\model_fm\\fm_au_alex_no.pth'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)

# 训练模型
for epoch in range(100):
    running_loss = 0.0
    correct = 0
    total = 0
    print('Epoch {}/{}'.format(epoch, 100))
    model.train()
    for data in EMA_train_dataloader:
        vision_inputs, touch_inputs, labels = data
        if torch.cuda.is_available():
            vision_inputs = vision_inputs.cuda()
            touch_inputs = touch_inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(vision_inputs ,touch_inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    train_acc = 100 * correct / total
    print(f'Train Accuracy of the model on the train images: {train_acc} %')
    train_acc_list.append(train_acc)
    torch.save(model.state_dict(), 'G:\\model_fm\\fm_au_alex_no.pth')

    correct = 0
    total = 0
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in EMA_test_dataloader:
            vision_inputs, touch_inputs, labels = data
            if torch.cuda.is_available():
                vision_inputs = vision_inputs.cuda()
                touch_inputs = touch_inputs.cuda()
                labels = labels.cuda()
            outputs = model(vision_inputs, touch_inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f'Test Accuracy of the model on the test images: {test_acc} %')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    test_acc_list.append(test_acc)
    test_precision = precision_score(all_labels, all_preds, average='weighted')
    test_recall = recall_score(all_labels, all_preds, average='weighted')
    test_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Test Precision: {test_precision}')
    print(f'Test Recall: {test_recall}')
    print(f'Test F1 Score: {test_f1}')

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 设置保存目录和文件名
    save_dir = 'G:\\picture'  # 指定保存图像的目标目录
    if not os.path.exists(save_dir):  # 如果目录不存在，则创建
        os.makedirs(save_dir)

    filename = f'Confusion_Matrix_{epoch + 1}.png'
    full_path = os.path.join(save_dir, filename)  # 完整的保存路径

    # 可视化混淆矩阵
    plt.rcParams.update({'font.size': 16})  # 减小全局字体大小
    plt.figure(figsize=(40, 32))  # 增加图像尺寸以适应更多类别

    # 创建热图并调整相关设置
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(63), yticklabels=range(63),
                annot_kws={"size": 8})  # 减小注释文本大小

    # 设置X轴标签旋转90度
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.title('Confusion Matrix', fontsize=24)

    # 在图中添加准确率文本
    plt.text(0.5, -0.1, f'Accuracy: {test_acc:.4f}', ha='center', va='center', fontsize=20,
             transform=plt.gca().transAxes)

    # 保存图像并显示
    plt.savefig(full_path, bbox_inches='tight')  # 使用完整路径保存图像

    print('Finished Training')


print('Finished Training')



end_time = time.time()
elapsed_time = end_time - start_time
print('Total training time: {:.2f} seconds'.format(elapsed_time))

plt.figure(figsize=(12, 6))
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch\nTotal training time: {:.2f} seconds'.format(elapsed_time))
plt.legend()
plt.grid(True)
plt.show()
