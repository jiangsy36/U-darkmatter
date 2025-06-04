import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import classification_report
import os
from tqdm import tqdm
import cv2
import gc

from train_pytorch import UNet, SegmentationDataset, create_masks, create_labels, visualize_results

def resize_images(images, target_size=(128, 128)):
    """
    调整图像大小
    
    参数:
        images: 形状为 (n_images, height, width) 的图像数组
        target_size: 目标大小 (height, width)
    
    返回:
        调整大小后的图像数组
    """
    n_images = len(images)
    resized = np.zeros((n_images, *target_size))
    for i in range(n_images):
        resized[i] = cv2.resize(images[i], target_size, interpolation=cv2.INTER_NEAREST)
    return resized

def load_and_evaluate_model(model_path, test_images_path, save_dir='evaluation_results'):
    """
    加载预训练的UNet模型并评估其性能
    
    参数:
        model_path: 预训练模型路径
        test_images_path: 测试图像路径
        save_dir: 评估结果保存目录
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载测试数据
    print('加载测试数据...')
    test_images_original = np.load(test_images_path)
    print(f'原始测试图像形状: {test_images_original.shape}')
    
    # 调整图像大小
    target_size = (128, 128)
    print(f'调整图像大小到 {target_size}...')
    test_images_original = resize_images(test_images_original, target_size)
    print(f'调整后的测试图像形状: {test_images_original.shape}')
    
    # 创建测试图像掩码
    print('创建测试图像掩码...')
    test_masks = create_masks(test_images_original)
    print(f'测试掩码形状: {test_masks.shape}')
    
    # 创建测试图像标签
    print('创建测试图像标签...')
    test_labels = create_labels(test_images_original)
    print(f'测试标签形状: {test_labels.shape}')
    
    # 准备数据
    test_images_all = test_images_original[..., np.newaxis]
    test_images_all = np.transpose(test_images_all, (0, 3, 1, 2))
    test_images_all = (test_images_all - np.min(test_images_all)) / (np.max(test_images_all) - np.min(test_images_all))
    print(f'预处理后的测试图像形状: {test_images_all.shape}')
    
    # 初始化模型
    model = UNet(n_channels=1, n_classes=4).to(device)
    
    # 加载预训练权重
    print(f'加载预训练模型: {model_path}')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 评估模型
    print("\n评估测试集...")
    test_accuracy, test_predictions, test_report = evaluate_model(model, test_images_all, test_masks, device)
    print(f'测试集准确率: {test_accuracy*100:.2f}%')
    print('\n测试集分类报告:')
    print(test_report)
    
    # 保存评估报告
    with open(os.path.join(save_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(f'测试集准确率: {test_accuracy*100:.2f}%\n\n')
        f.write('测试集分类报告:\n')
        f.write(test_report)
    
    # 可视化结果
    for ix in range(len(test_images_all)):
        visualize_results(
            test_images_all[ix],
            test_masks[ix],
            test_predictions[ix],
            save_path=os.path.join(save_dir, f'Test_{ix+1}.png'),
            title_prefix='Test'
        )
    
    # 清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return test_accuracy, test_predictions, test_report

def evaluate_model(model, test_images, test_masks, device, batch_size=4):
    """
    评估模型性能
    
    参数:
        model: UNet模型
        test_images: 测试图像
        test_masks: 测试掩码
        device: 计算设备
        batch_size: 批次大小
    """
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

if __name__ == '__main__':
    # 设置路径
    model_path = 'training_results/unet_model.pth'
    test_images_path = '../Datasets/test_data.npy'
    
    # 加载和评估模型
    test_accuracy, test_predictions, test_report = load_and_evaluate_model(
        model_path=model_path,
        test_images_path=test_images_path
    ) 