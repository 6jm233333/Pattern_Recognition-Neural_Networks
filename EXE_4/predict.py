import os
import json
import time

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # 1. 设置图片文件夹路径
    img_dir = "/data1/JiamingLiu/模式识别/Flower/flower_photos/tulips"
    assert os.path.exists(img_dir), f"file: '{img_dir}' does not exist."

    # 2. 读取类别索引
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 3. 创建模型并加载权重
    model_name = "vit_base_patch16_224_in21k"
    model = create_model(num_classes=5, has_logits=False).to(device)
    model_weight_path = "/data1/JiamingLiu/模式识别/Code/Exe_4/weights/model-49.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 4. 批量处理图片
    save_dir = "/data1/JiamingLiu/模式识别/Code/Exe_4/pic"
    os.makedirs(save_dir, exist_ok=True)
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    img_files = img_files[:16]  # 只处理16张

    # 创建4x4子图
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(f"{model_name} predictions", fontsize=20)

    for idx, img_file in enumerate(img_files):
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = data_transform(img)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

        with torch.no_grad():
            output = torch.squeeze(model(img_tensor.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())

        plt.figure()
        plt.imshow(img)
        plt.title(print_res)
        plt.axis('off')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        single_save_name = f"{model_name}_{timestamp}_{img_file}"
        single_save_path = os.path.join(save_dir, single_save_name)
        plt.savefig(single_save_path)
        plt.close()
        time.sleep(1)  # 保证时间戳不同

        # --------- 画到大图的子图上 ---------
        ax = axes[idx // 4, idx % 4]
        ax.imshow(img)
        ax.set_title(print_res, fontsize=10)
        ax.axis('off')

    # 保存大图，文件名为模型名+时间戳
    batch_timestamp = time.strftime("%Y%m%d_%H%M%S")
    batch_save_name = f"大图_{model_name}_{batch_timestamp}_batch.png"
    batch_save_path = os.path.join(save_dir, batch_save_name)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出总标题空间
    plt.savefig(batch_save_path)
    plt.close()



if __name__ == '__main__':
    main()