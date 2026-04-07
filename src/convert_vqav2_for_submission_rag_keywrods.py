import os
import argparse
import json
from sklearn.metrics import precision_recall_curve
import sys
import torch
import re
from itertools import product


def get_label_type(label, label_dic):
    """根据标签类型分类并统计数量"""
    if label in ('红一恶意'):
        tmp = 0
    elif label in ('政治恶意'):
        tmp = 1
    elif label in ('社会恶意'):
        tmp = 2
    elif label in ('色情恶意'):
        tmp = 3
    elif label in ('违法恶意'):
        tmp = 4
    else:
        tmp = 6
    
    label_dic[tmp] = label_dic.get(tmp, 0) + 1
    return tmp


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VQA评估脚本')
    parser.add_argument('--mejson', type=str, default="", help='输入JSON文件路径')
    parser.add_argument('--split', type=str, required=True, help='测试集分割文件路径')
    return parser.parse_args()


def load_json_file(file_path):
    """安全加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"错误加载文件 {file_path}: {e}")
        return None


def load_results_file(file_path):
    """加载结果文件并处理错误行"""
    results = []
    error_line = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_idx, line in enumerate(f):
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    error_line += 1
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
        return [], 0
    
    return results, error_line


def calculate_metrics(confusion_matrix):
    """计算准确率、召回率和F1分数"""
    eps = 1e-9
    metrics = []
    
    for i in range(confusion_matrix.size()[0]):
        p = confusion_matrix[i, i].item() / (confusion_matrix[i, :].sum().item() + eps)
        r = confusion_matrix[i, i].item() / (confusion_matrix[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        metrics.append((p, r, f1))
    
    return metrics


if __name__ == '__main__':
    args = parse_args()
    
    # 确定输入文件路径
    src = args.mejson
    
    print(f"输入文件: {src}")
    print(f"测试分割文件: {args.split}")
    
    # 加载结果文件
    resultsori, error_line = load_results_file(src)
    if not resultsori:
        print("无法加载结果文件，程序退出")
        sys.exit(1)
    
    # 处理结果数据
    results = {x['question_id']: float(x['score']) for x in resultsori}
    resultstext = {x['question_id']: x['text'] for x in resultsori}
    resultprompt = {x['question_id']: x['prompt'] for x in resultsori}
    
    # 加载测试分割文件
    test_split = load_json_file(args.split)
    if test_split is None:
        print("无法加载测试分割文件，程序退出")
        sys.exit(1)
    
    split_ids = set([x['id'] for x in test_split])
    datalist = {x['id']: x for x in test_split}
    
    print(f'总结果数: {len(results)}, 总分割数: {len(test_split)}, 错误行数: {error_line}')
    
    # 初始化变量
    all_answers = []
    y_scores = []
    y_id = []
    y_true = []
    y_audit_label = []
    
    gt_label_dic = {i: 0 for i in range(7)}
    num_label = 2
    
    # 处理测试数据
    for x in test_split:
        if x['id'] not in results:
            continue
            
        all_answers.append({
            'question_id': x['id'],
            'answer': ''
        })
        
        score = results[x['id']]
        
        # 确定标签
        if 'label' in x:
            label = str(x['label'])
            if '违法' in label:
                continue
        elif 'output' in x:
            label = str(x['output'])
            if '是' in label:
                label = '是'
            else:
                label = '否'
        else:
            continue
        
        # 确定二进制标签
        lab = 1 if ('是' in label or '舆情' in label) else 0
        
        y_true.append(lab)
        y_scores.append(score)
        y_audit_label.append(label)
        y_id.append(x['id'])
        
        get_label_type(label, gt_label_dic)
    
    print(f"正样本数量: {sum(y_true)}, 负样本数量: {len(y_true) - sum(y_true)}, 总样本数: {len(y_true)}")
    print(f"真实标签分布: {gt_label_dic}")
    
    # 计算混淆矩阵
    confusion = torch.zeros(num_label, num_label, dtype=torch.long)
    
    for i, score in enumerate(y_scores):
        id_val = y_id[i]
        text = resultstext[id_val]
        
        # 预测标签
        pre_label = 1 if ('是' in text[:2] or '恶意' in text[:2]) else 0
        confusion[pre_label, y_true[i]] += 1
    
    # 输出结果
    print("混淆矩阵:")
    print(confusion)
    
    print("准确率、召回率和F1分数:")
    metrics = calculate_metrics(confusion)
    for i, (p, r, f1) in enumerate(metrics):
        print("标签 {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
