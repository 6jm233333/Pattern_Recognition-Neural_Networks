#coding=utf-8
"""
 主程序：主要完成四个功能
（1）训练：定义网络，损失函数，优化器，进行训练，生成模型
（2）验证：验证模型准确率
（3）测试：测试模型在测试集上的准确率
（4）help：打印log信息

"""

from config import opt
import os
import matplotlib.pyplot as plt
import numpy as np
#import models
import torch as t
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from torch.autograd import Variable
from torchvision import models, transforms
from torch import nn
import time
import csv
from torchvision.utils import save_image
device = t.device("cuda" if opt.use_gpu else "cpu")

"""模型训练：定义网络，定义数据，定义损失函数和优化器，训练并计算指标，计算在验证集上的准确率"""
def train(**kwargs):
    """根据命令行参数更新配置"""
    import pandas as pd
    opt.parse(kwargs)
    # vis = Visualizer(opt.env)
    vis = None
    device = t.device("cuda" if opt.use_gpu else "cpu")

    # (1)step1：加载网络，若有预训练模型也加载
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 2)
    model.to(device)

    # (2)step2：处理数据
    train_data = DogCat(opt.train_data_root, train=True)  # 训练集
    val_data = DogCat(opt.train_data_root, train=False)   # 验证集

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # (3)step3：定义损失函数和优化器
    criterion = t.nn.CrossEntropyLoss()  # 交叉熵损失
    lr = opt.lr  # 学习率
    optimizer = t.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = float('inf')

    # 新增：训练过程数据记录
    train_losses, val_losses, val_accuracies = [], [], []

    # 新增：训练结果保存目录
    save_root = "/data1/JiamingLiu/模式识别/Code/Exe_3/dogsVScats/train_pic"
    os.makedirs(save_root, exist_ok=True)

    # (5)开始训练
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()
        model.train()

        for ii, (data, label) in enumerate(train_dataloader):

            # 训练模型参数
            input = Variable(data)
            target = Variable(label)

            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            # 梯度清零
            optimizer.zero_grad()
            score = model(input)

            loss = criterion(score, target)
            loss.backward()  # 反向传播

            # 更新参数
            optimizer.step()

            # 更新统计指标及可视化
            loss_meter.add(loss.item())
            confusion_matrix.add(score.detach(), target.detach())

            if ii % opt.print_freq == opt.print_freq - 1:
                if vis is not None:
                    vis.plot('loss', loss_meter.value()[0])

                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        # 计算验证集上的指标及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        # 记录损失和准确率
        train_losses.append(loss_meter.value()[0])
        val_accuracies.append(val_accuracy)
        # 验证损失（可选：这里用训练损失代替，如需精确可在val函数中返回val_loss）
        val_losses.append(loss_meter.value()[0])

        # 以模型名创建文件夹
        model_name = time.strftime('model' + '%m%d_%H:%M:%S')
        model_dir = os.path.join(save_root, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # 保存模型
        model_path = os.path.join(model_dir, model_name + ".pth")
        t.save(model.state_dict(), model_path)

        if vis is not None:
            vis.plot('val_accuracy', val_accuracy)
        log_str = "epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0],
            val_cm=str(val_cm.value()),
            train_cm=str(confusion_matrix.value()),
            lr=lr
        )
        print(log_str)  # 打印到控制台
        if vis is not None:  # 只有vis存在时才调用log
            vis.log(log_str)

        print("epoch:", epoch, "loss:", loss_meter.value()[0], "accuracy:", val_accuracy)

        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        previous_loss = loss_meter.value()[0]

    # === 训练结束后，保存曲线和csv ===

    # 保存准确率曲线
    plt.figure()
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(model_dir, "val_accuracy_curve.png"))
    plt.close()

    # 保存损失曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train/Val Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(model_dir, "loss_curve.png"))
    plt.close()

    # 保存csv报告
    df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses)+1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies
    })
    df.to_csv(os.path.join(model_dir, "train_report.csv"), index=False)
		

"""计算模型在验证集上的准确率等信息"""
@t.no_grad()
def val(model,dataloader):

	model.eval() #将模型设置为验证模式

	confusion_matrix = meter.ConfusionMeter(2)
	for ii,data in enumerate(dataloader):
		input,label = data
		# val_input = Variable(input,volatile=True)
		# val_label = Variable(label.long(),volatile=True)
		with t.no_grad():
			val_input = input.to(device)  # device
			val_label = label.long().to(device)
		if opt.use_gpu:
			val_input = val_input.cuda()
			val_label = val_label.cuda()

		score = model(val_input)
		confusion_matrix.add(score.detach().squeeze(),label.long())

	model.train() #模型恢复为训练模式
	cm_value = confusion_matrix.value()
	accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())

	return confusion_matrix,accuracy

""""""
def test(**kwargs):
    opt.parse(kwargs)

    # 获取模型名
    if opt.load_model_path:
        model_name = os.path.splitext(os.path.basename(opt.load_model_path))[0]
    else:
        model_name = "untrained"

    # 创建保存目录
    save_dir = os.path.join("/data1/JiamingLiu/模式识别/Code/Exe_3/dogsVScats/test_pic", model_name)
    os.makedirs(save_dir, exist_ok=True)

    # data
    test_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []

    # model
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(t.load(opt.load_model_path, map_location=device))
    model.to(device)
    model.eval()

    # 反归一化（如果有用 transforms.Normalize）
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    img_save_count = 0  # 只保存前N张图片
    max_save_imgs = 10  # 可调整

    for ii, (data, path) in enumerate(test_dataloader):
        input = data.to(device)
        score = model(input)
        _, predicted = t.max(score.data, 1)
        predicted = predicted.data.cpu().numpy().tolist()
        # path 可能是图片路径或编号
        if isinstance(path, t.Tensor):
            path = path.numpy().tolist()
        for b, (i, j) in enumerate(zip(path, predicted)):
            label = "Dog" if j == 1 else "Cat"
            results.append([i, label])
            # 保存部分图片
            if img_save_count < max_save_imgs:
                img_tensor = input[b].cpu()
                # 反归一化
                img_tensor = inv_normalize(img_tensor)
                img_tensor = t.clamp(img_tensor, 0, 1)
                img_name = os.path.basename(str(i))
                save_path = os.path.join(save_dir, f"{img_name}_{label}.jpg")
                save_image(img_tensor, save_path)
                img_save_count += 1

    # 保存csv
    csv_path = os.path.join(save_dir, "result.csv")
    write_csv(results, csv_path)
    print(f"测试结果和部分图片已保存到: {save_dir}")
    return results


""""""



def write_csv(results,file_name):
	with open(file_name,"w") as f:
		writer = csv.writer(f)
		writer.writerow(['id','label'])
		writer.writerows(results)

def validate(**kwargs):
    opt.parse(kwargs)
    # 构建模型
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(512, 2)
    if opt.load_model_path:
        model.load_state_dict(t.load(opt.load_model_path, map_location=device))
        # 取出pth文件名（不带扩展名）
        pth_name = os.path.splitext(os.path.basename(opt.load_model_path))[0]
    else:
        pth_name = "untrained"
    model.to(device)
    # 构建 dataloader
    val_data = DogCat(opt.train_data_root, train=False)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    # 调用 val
    val_cm, val_accuracy = val(model, val_dataloader)
    print("验证集混淆矩阵：\n", val_cm.value())
    print("验证集准确率：", val_accuracy)

    # 1. 创建保存图片的目录
    save_dir = f"/data1/JiamingLiu/模式识别/Code/Exe_3/dogsVScats/picture/{pth_name}"
    os.makedirs(save_dir, exist_ok=True)

    # 2. 保存混淆矩阵图片
    cm = val_cm.value()
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Cat', 'Dog']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # 标注数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # 3. 保存部分预测结果图片
    # 需要 val_data 支持 __getitem__ 返回图片路径或PIL对象
    images, labels, preds = [], [], []
    model.eval()
    with t.no_grad():
        for i, (data, label) in enumerate(val_dataloader):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            labels.extend(label.numpy())
            # 假设 val_data[i] 返回 (img, label)，img为PIL或tensor
            for j in range(data.size(0)):
                img = val_data[i * opt.batch_size + j][0]
                images.append(img)
            if len(images) >= 16:
                break
    # 画前16张
    plt.figure(figsize=(12, 12))
    for idx in range(16):
        plt.subplot(4, 4, idx+1)
        img = images[idx]
        # 如果是tensor，转为numpy
        if isinstance(img, t.Tensor):
            img = img.permute(1,2,0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"True:{labels[idx]} Pred:{preds[idx]}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sample_predictions.png"))
    plt.close()

    return val_accuracy

import fire
if __name__ == '__main__':
    # 方法1：字典式暴露
    fire.Fire({
        "train": train,
        "validate": validate,  # 用 validate 包装
        "test": test,
        "help": lambda: print("可用命令: train, validate, test")
    })




