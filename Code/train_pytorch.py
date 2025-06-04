import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
import matplotlib.patches as mpatches
from sklearn.metrics import classification_report
import cv2
import gc

def create_masks(images):
    """
    为图像创建掩码，将像素值分为4类
    返回形状为 (n_images, height, width) 的掩码，值为类别索引 [0,1,2,3]
    """
    n_images = len(images)
    height, width = images.shape[1], images.shape[2]
    masks = np.zeros((n_images, height, width), dtype=np.int64)
    
    # 收集所有像素值并分析分布
    all_pixel_values = []
    for ix in range(n_images):
        all_pixel_values.extend(images[ix].flatten())
    
    all_pixel_values = np.array(all_pixel_values)
    
    # 分析数据分布
    min_val = np.min(all_pixel_values)
    max_val = np.max(all_pixel_values)
    mean_val = np.mean(all_pixel_values)
    std_val = np.std(all_pixel_values)
    
    # 设置固定的背景阈值
    background_threshold = 525  # 基准值
    background_tolerance = 0.001  # 容差范围
    
    # 对于非背景像素，计算分位数
    non_background_pixels = all_pixel_values[np.abs(all_pixel_values - background_threshold) > background_tolerance]
    if len(non_background_pixels) > 0:
        p33, p66 = np.percentile(non_background_pixels, [33, 66])
    else:
        p33, p66 = mean_val, mean_val + std_val
    
    print(f"\n数据分布统计:")
    print(f"最小值: {min_val:.2e}")
    print(f"最大值: {max_val:.2e}")
    print(f"均值: {mean_val:.2e}")
    print(f"标准差: {std_val:.2e}")
    print(f"背景基准值: {background_threshold:.2e}")
    print(f"背景容差范围: ±{background_tolerance:.2e}")
    print(f"33%分位数: {p33:.2e}")
    print(f"66%分位数: {p66:.2e}")
    
    # 计算每个区间的像素数量
    hist, bins = np.histogram(all_pixel_values, bins=50)
    print("\n像素值分布直方图的前10个区间:")
    for i in range(min(10, len(hist))):
        print(f"区间 {bins[i]:.2e} - {bins[i+1]:.2e}: {hist[i]} 像素")
    
    class_counts = np.zeros(4)  # 用于统计每个类别的像素数量
    
    for ix in range(n_images):
        image = images[ix]
        
        # 使用固定阈值策略划分类别
        # 背景类：在基准值附近的像素
        masks[ix][np.abs(image - background_threshold) <= background_tolerance] = 0
        
        # 其他类别
        non_background_mask = np.abs(image - background_threshold) > background_tolerance
        masks[ix][non_background_mask & (image <= p33)] = 1  # 低强度
        masks[ix][non_background_mask & (image > p33) & (image <= p66)] = 2  # 中强度
        masks[ix][non_background_mask & (image > p66)] = 3  # 高强度
        
        # 统计每个类别的像素数量
        for c in range(4):
            class_counts[c] += np.sum(masks[ix] == c)
    
    # 打印类别分布信息
    total_pixels = np.sum(class_counts)
    print("\n类别分布统计:")
    class_names = ['背景', '低强度', '中强度', '高强度']
    for c in range(4):
        percentage = (class_counts[c] / total_pixels) * 100
        print(f"类别 {c} ({class_names[c]}): {class_counts[c]:.0f} 像素 ({percentage:.2f}%)")
    
    # 验证掩码中的类别
    unique_classes = np.unique(masks)
    print(f"\n掩码中的类别: {unique_classes}")
    
    # 检查类别分布
    min_class_percentage = 5  # 设置最小期望百分比
    imbalanced_classes = []
    for c in range(4):
        percentage = (class_counts[c] / total_pixels) * 100
        if percentage < min_class_percentage:
            imbalanced_classes.append(f"类别{c}({class_names[c]}): {percentage:.2f}%")


    return masks

def create_labels(images):
    """
    为图像创建RGB标签
    返回形状为 (n_images, height, width, 3) 的标签
    """
    n_images = len(images)
    print(n_images)
    height, width = images.shape[1], images.shape[2]
    labels = np.zeros((n_images, height, width, 3))
    
    # 定义颜色映射
    color_map = {
        0: [0, 0, 0],
        1: [0.35, 1, 0.25],
        2: [0, 0.5, 1],
        3: [1, 0.2, 0.2]
    }
    
    for ix in range(n_images):
        for class_idx, color in color_map.items():
            mask = images[ix] == class_idx
            for c in range(3):
                print(c)
                labels[ix, :, :, c][mask] = color[c]
    
    return labels

# 自定义数据集类
class SegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = torch.from_numpy(images).float()
        self.masks = torch.from_numpy(masks)  # 不需要转换为float
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        return image, mask

# U-Net的双卷积块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# U-Net模型
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器路径
        x = self.up1(x5)
        x = self.conv_up1(torch.cat([x4, x], dim=1))
        
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x3, x], dim=1))
        
        x = self.up3(x)
        x = self.conv_up3(torch.cat([x2, x], dim=1))
        
        x = self.up4(x)
        x = self.conv_up4(torch.cat([x1, x], dim=1))
        
        x = self.outc(x)
        return self.softmax(x)

# 加载数据
def load_data():
    print('加载数据...')
    images_original = np.load('../Datasets/training_data.npy')
    test_images_original = np.load('../Datasets/test_data.npy')
    
    # 打印原始数据的基本信息
    print(f"\n原始数据统计:")
    print(f"训练集形状: {images_original.shape}")
    print(f"测试集形状: {test_images_original.shape}")
    print(f"训练集值范围: [{np.min(images_original)}, {np.max(images_original)}]")
    print(f"测试集值范围: [{np.min(test_images_original)}, {np.max(test_images_original)}]")
    
    # 调整图像大小以减少内存使用
    target_size = (128, 128)  # 减小图像尺寸
    
    def resize_images(images):
        n_images = len(images)
        resized = np.zeros((n_images, *target_size))
        for i in range(n_images):
            resized[i] = cv2.resize(images[i], target_size, interpolation=cv2.INTER_NEAREST)
        return resized
    
    print('\n调整图像大小...')
    images_original = resize_images(images_original)
    test_images_original = resize_images(test_images_original)
    
    print('\n创建训练图像掩码...')
    masks = create_masks(images_original)  # 在归一化之前创建掩码
    print('\n创建测试图像掩码...')
    test_masks = create_masks(test_images_original)  # 在归一化之前创建掩码
    
    print('\n创建训练图像标签...')
    labels = create_labels(images_original)
    print('\n创建测试图像标签...')
    test_labels = create_labels(test_images_original)
    
    # 添加通道维度并转换为PyTorch格式
    images_all = images_original[..., np.newaxis]
    test_images_all = test_images_original[..., np.newaxis]
    
    # 转换维度顺序为PyTorch格式 (N, C, H, W)
    images_all = np.transpose(images_all, (0, 3, 1, 2))
    test_images_all = np.transpose(test_images_all, (0, 3, 1, 2))
    
    # 数据归一化 - 在创建掩码之后进行
    images_all = (images_all - np.min(images_all)) / (np.max(images_all) - np.min(images_all))
    test_images_all = (test_images_all - np.min(test_images_all)) / (np.max(test_images_all) - np.min(test_images_all))
    
    return images_all, test_images_all, masks, test_masks, labels, test_labels

# 训练函数
def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    history = {
        'loss': [],
        'accuracy': []
    }
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_correct = 0
        total_pixels = 0
        
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device)
            masks = masks.long().to(device)  # [B, H, W]
            
            optimizer.zero_grad()
            outputs = model(images)  # [B, C, H, W]
            
            # 修正形状不匹配问题
            B, C, H, W = outputs.shape
            outputs = outputs.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            masks = masks.reshape(-1)  # [B*H*W]
            
            loss = criterion(outputs, masks)
            
            if torch.isnan(loss):
                print("警告：损失值为NaN！")
                continue
                
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            pred_masks = torch.argmax(outputs, dim=1)
            correct = (pred_masks == masks).sum().item()
            total_correct += correct
            total_pixels += masks.numel()
            
            running_loss += loss.item()
            
        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = total_correct / total_pixels * 100
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)
        
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    
    return history

def plot_training_history(history, save_dir='.'):
    """
    绘制训练历史
    """
    # 创建图形
    plt.figure(figsize=(15, 5))
    
    # 绘制损失值
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], 'b-', label='Training Loss')
    plt.title('Training Loss Over Time', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], 'r-', label='Training Accuracy')
    plt.title('Training Accuracy Over Time', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.show()

# 评估函数
def evaluate_model(model, test_images, test_masks, device, batch_size=4):
    model.eval()
    total_correct = 0
    total_pixels = 0
    all_preds = []
    all_true = []
    
    # 批量处理数据
    n_samples = len(test_images)
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        with torch.no_grad():
            # 处理当前批次
            images_batch = torch.from_numpy(test_images[i:batch_end]).float().to(device)
            masks_batch = torch.from_numpy(test_masks[i:batch_end]).long().to(device)
            
            outputs = model(images_batch)
            
            # 计算预测
            B, C, H, W = outputs.shape
            outputs_reshaped = outputs.permute(0, 2, 3, 1).reshape(-1, C)
            pred_masks = torch.argmax(outputs_reshaped, dim=1)
            masks_flat = masks_batch.reshape(-1)
            
            # 统计正确预测
            correct = (pred_masks == masks_flat).sum().item()
            total_correct += correct
            total_pixels += masks_flat.numel()
            
            # 收集预测结果
            pred_masks = pred_masks.reshape(B, H, W)
            all_preds.append(pred_masks.cpu().numpy())
            all_true.append(masks_flat.cpu().numpy())
            
            # 清理GPU内存
            del images_batch, masks_batch, outputs, pred_masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 计算准确率
    accuracy = total_correct / total_pixels
    
    # 合并所有预测结果
    all_preds = np.concatenate([p.reshape(-1) for p in all_preds])
    all_true = np.concatenate(all_true)
    
    # 确保所有类别都存在于标签中
    unique_labels = np.unique(np.concatenate([all_true, all_preds]))
    print(f"预测中的唯一类别: {unique_labels}")
    
    # 生成分类报告
    number_to_class = ['background', 'glowing', 'hot pixel', 'cluster']
    labels = list(range(len(number_to_class)))  # [0,1,2,3]
    report = classification_report(y_true=all_true, y_pred=all_preds,
                                target_names=number_to_class,
                                labels=labels,
                                zero_division=0)
    
    # 重塑预测结果为原始形状
    predictions = all_preds.reshape(n_samples, H, W)
    
    return accuracy, predictions, report

def visualize_results(image, true_mask, pred_mask, save_path=None, title_prefix=''):
    """
    image: 形状为(C, H, W)的图像
    true_mask: 形状为(H, W)的真实掩码
    pred_mask: 形状为(H, W)的预测掩码
    """
    # 创建图例
    red_patch = mpatches.Patch(color=[1, 0.2, 0.2], label='Cluster')
    blue_patch = mpatches.Patch(color=[68/255 , 1/255, 84/255 ], label='Hot pixel')
    green_patch = mpatches.Patch(color=[1, 1, 0], label='Glowing')
    black_patch = mpatches.Patch(color=[32 / 255, 144 / 255, 140 / 255], label='Background')
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    
    # 显示原始图像
    ax[0].imshow(np.squeeze(image.transpose(1, 2, 0)), cmap="gray")
    ax[0].set_title(f'{title_prefix} Image', fontsize=25)
    ax[0].set_xlabel('pixels', fontsize=16)
    ax[0].set_ylabel('pixels', fontsize=16)
    
    # 显示真实标签
    ax[1].imshow(true_mask)
    ax[1].set_title('True Label', fontsize=25)
    ax[1].set_xlabel('pixels', fontsize=16)
    ax[1].set_ylabel('pixels', fontsize=16)
    
    # 显示预测标签
    ax[2].imshow(pred_mask)
    ax[2].set_title('Predicted Label', fontsize=25)
    ax[2].set_xlabel('pixels', fontsize=16)
    ax[2].set_ylabel('pixels', fontsize=16)
    
    plt.legend(loc='upper center', bbox_to_anchor=(-0.12, -0.15), fontsize=18,
              handles=[red_patch, blue_patch, green_patch, black_patch], ncol=4)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize(save_path=None, title_prefix=''):
    """
    image: 形状为(C, H, W)的图像
    true_mask: 形状为(H, W)的真实掩码
    pred_mask: 形状为(H, W)的预测掩码
    """
    # 加载数据
    image, test_images_all, true_mask, test_masks, labels, test_labels = load_data()
    ix_random_training = random.randint(0, len(image) - 1)
    # 创建图例
    red_patch = mpatches.Patch(color=[1, 0.2, 0.2], label='Cluster')
    blue_patch = mpatches.Patch(color=[68 / 255, 1 / 255, 84 / 255], label='Hot pixel')
    green_patch = mpatches.Patch(color=[1, 1, 0], label='Glowing')
    black_patch = mpatches.Patch(color=[32 / 255, 144 / 255, 140 / 255], label='Background')

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # 显示原始图像
    ax[0].imshow(np.squeeze(image[ix_random_training].transpose(1, 2, 0)), cmap="gray")
    ax[0].set_title(f'{title_prefix} Image', fontsize=25)
    ax[0].set_xlabel('pixels', fontsize=16)
    ax[0].set_ylabel('pixels', fontsize=16)

    # 显示真实标签
    ax[1].imshow(true_mask[ix_random_training])
    ax[1].set_title('True Label', fontsize=25)
    ax[1].set_xlabel('pixels', fontsize=16)
    ax[1].set_ylabel('pixels', fontsize=16)


    plt.legend(loc='upper center', bbox_to_anchor=(-0.12, -0.15), fontsize=18,
               handles=[red_patch, blue_patch, green_patch, black_patch], ncol=4)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 超参数
    BATCH_SIZE =1
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # 创建保存目录
    save_dir = 'training_results'
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    images_all, test_images_all, masks, test_masks, labels, test_labels = load_data()
    
    # 打印数据信息
    print(f"训练图像形状: {images_all.shape}")
    print(f"训练掩码形状: {masks.shape}")
    print(f"掩码中的唯一值: {np.unique(masks)}")
    print(f"训练图像值范围: [{images_all.min():.2f}, {images_all.max():.2f}]")
    
    # 验证数据
    assert np.all(masks >= 0) and np.all(masks < 4), "掩码值超出范围！"
    assert np.all(images_all >= 0) and np.all(images_all <= 1), "图像值未正确归一化！"
    
    # 验证每个类别是否都存在
    unique_classes = np.unique(masks)

    
    # 创建数据加载器
    train_dataset = SegmentationDataset(images_all, masks)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 初始化模型
    model = UNet(n_channels=1, n_classes=4).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练模型并获取历史记录
    history = train_model(model, train_loader, criterion, optimizer, device, EPOCHS)
    
    # 绘制训练历史
    plot_training_history(history, save_dir)
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_dir, 'unet_model.pth'))
    
    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 评估模型
    print("\n评估训练集...")
    train_accuracy, train_predictions, train_report = evaluate_model(model, images_all, masks, device, BATCH_SIZE)
    print(f'训练集准确率: {train_accuracy*100:.2f}%')
    print('\n训练集分类报告:')
    print(train_report)
    
    print("\n评估测试集...")
    test_accuracy, test_predictions, test_report = evaluate_model(model, test_images_all, test_masks, device, BATCH_SIZE)
    print(f'测试集准确率: {test_accuracy*100:.2f}%')
    print('\n测试集分类报告:')
    print(test_report)
    
    # 可视化结果
    # 随机训练样本
    ix_random_training = random.randint(0, len(images_all)-1)
    visualize_results(
        images_all[ix_random_training],
        masks[ix_random_training],
        train_predictions[ix_random_training],
        save_path=os.path.join(save_dir, f'Train_{ix_random_training+1}.png'),
        title_prefix='Training'
    )
    
    # 测试样本
    for ix in range(len(test_images_all)):
        visualize_results(
            test_images_all[ix],
            test_masks[ix],
            test_predictions[ix],
            save_path=os.path.join(save_dir, f'Test_{ix+1}.png'),
            title_prefix='Test'
        )

if __name__ == '__main__':
    main()
    visualize()
