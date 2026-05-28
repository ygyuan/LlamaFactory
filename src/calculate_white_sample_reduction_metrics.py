#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白样本降量指标计算脚本
专门用于计算白样本的准确率、召回率和F1指标
只使用score的第一个浮点数（白样本概率）进行计算
"""

import os
import json
import argparse
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix


def load_jsonl_file(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def extract_predictions_and_labels(pred_file, test_file):
    """
    从预测文件和测试文件中提取预测结果和真实标签
    
    Args:
        pred_file: 预测结果文件路径
        test_file: 测试数据文件路径
        
    Returns:
        tuple: (predictions, ground_truth, conversations)
            predictions: 预测结果字典 {question_id: score}
            ground_truth: 真实标签字典 {question_id: label}
            conversations: 对话内容字典 {question_id: conversations}
    """
    # 加载预测结果
    pred_data = load_jsonl_file(pred_file)
    predictions = {}
    for item in pred_data:
        if 'question_id' in item and 'score' in item:
            predictions[item['question_id']] = item['score']
    
    # 加载测试数据
    test_data = load_jsonl_file(test_file)
    ground_truth = {}
    conversations = {}
    for item in test_data:
        if 'id' in item and 'label' in item:
            ground_truth[item['id']] = item['label']
            # 同步保存原始 conversations，便于后续 TN/FP 样本回溯
            if 'conversations' in item:
                conversations[item['id']] = item['conversations']
    
    return predictions, ground_truth, conversations


def convert_labels_to_binary(labels):
    """将标签转换为二分类格式（白样本=1，黑样本=0）"""
    binary_labels = []
    for label in labels:
        if label == '100' or label == '21000':  # 白样本
            binary_labels.append(1)
        else:  # 黑样本
            binary_labels.append(0)
    return binary_labels


def extract_white_probabilities(scores):
    """
    提取白样本概率（score的第一个浮点数）
    
    Args:
        scores: 预测分数列表
        
    Returns:
        list: 白样本概率列表
    """
    white_probs = []
    for score in scores:
        if isinstance(score, list) and len(score) >= 2:
            # 使用score的第一个浮点数（白样本概率）
            white_probs.append(float(score[0]))
        elif isinstance(score, (int, float)):
            # 如果是单个数值，直接使用
            white_probs.append(float(score))
        else:
            # 默认值
            white_probs.append(0.0)
    return white_probs


def calculate_white_sample_metrics(predictions, ground_truth):
    """
    计算白样本的precision和recall指标
    
    Args:
        predictions: 预测结果字典 {question_id: score}
        ground_truth: 真实标签字典 {question_id: label}
        
    Returns:
        dict: 包含白样本指标的字典
    """
    # 对齐预测结果和真实标签
    aligned_predictions = []
    aligned_truth = []
    sample_ids = []
    
    for sample_id in predictions:
        if sample_id in ground_truth:
            aligned_predictions.append(predictions[sample_id])
            aligned_truth.append(ground_truth[sample_id])
            sample_ids.append(sample_id)
    
    print(f"有效样本数量: {len(aligned_predictions)}")
    
    # 转换为二分类标签（白样本=1，黑样本=0）
    y_true_binary = convert_labels_to_binary(aligned_truth)
    
    # 提取白样本概率（score的第一个浮点数）
    y_scores_white = extract_white_probabilities(aligned_predictions)
    
    print(f"白样本概率范围: min={min(y_scores_white):.4f}, max={max(y_scores_white):.4f}, mean={np.mean(y_scores_white):.4f}")
    
    # 计算PR曲线 - 白样本作为正类（1）
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_scores_white)
    
    
    # 计算F1分数
    f1_scores = []
    for i in range(len(precision)):
        if precision[i] + recall[i] > 0:
            f1 = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1 = 0
        f1_scores.append(f1)
    
    # 找到最佳F1阈值
    best_f1_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_index] if best_f1_index < len(thresholds) else thresholds[-1]
    best_precision = precision[best_f1_index]
    best_recall = recall[best_f1_index]
    best_f1 = f1_scores[best_f1_index]
    
    # 使用最佳阈值进行预测
    y_pred_binary = [1 if score >= best_threshold else 0 for score in y_scores_white]
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    # 计算白样本相关指标
    # 注意：confusion_matrix的ravel顺序是 [TN, FP, FN, TP]
    tn, fp, fn, tp = cm.ravel()
    
    # 白样本总数
    white_samples_total = sum(y_true_binary)
    
    # 白样本精确率（Precision）
    white_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # 白样本召回率（Recall）
    white_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # 白样本F1分数
    white_f1 = 2 * white_precision * white_recall / (white_precision + white_recall) if (white_precision + white_recall) > 0 else 0
    
    # 白样本降量率（被正确识别为白样本的比例）
    white_reduction_rate = tp / white_samples_total if white_samples_total > 0 else 0
    
    # 误杀率（白样本被误判为黑样本的比例）
    false_negative_rate = fn / white_samples_total if white_samples_total > 0 else 0
    
    metrics = {
        'best_threshold': best_threshold,
        'white_precision': white_precision,
        'white_recall': white_recall,
        'white_f1': white_f1,
        'white_reduction_rate': white_reduction_rate,
        'false_negative_rate': false_negative_rate,
        'confusion_matrix': cm.tolist(),
        'white_samples_total': white_samples_total,
        'black_samples_total': len(y_true_binary) - white_samples_total,
        'true_positives': tp,
        'false_negatives': fn,
        'false_positives': fp,
        'true_negatives': tn
    }
    
    return metrics


def print_white_sample_metrics(metrics):
    """打印白样本指标结果"""
    print("\n" + "="*60)
    print("白样本指标计算结果")
    print("="*60)
    print("使用score的第一个浮点数（白样本概率）进行计算")
    
    print(f"\n最佳阈值: {metrics['best_threshold']:.6f}")
    
    print("\n核心白样本指标:")
    print(f"白样本精确率 (Precision): {metrics['white_precision']:.4f}")
    print(f"白样本召回率 (Recall): {metrics['white_recall']:.4f}")
    print(f"白样本F1分数: {metrics['white_f1']:.4f}")
    
    print("\n降量相关指标:")
    print(f"白样本降量率: {metrics['white_reduction_rate']:.4f} ({metrics['true_positives']}/{metrics['white_samples_total']})")
    print(f"误杀率: {metrics['false_negative_rate']:.4f} ({metrics['false_negatives']}/{metrics['white_samples_total']})")
    
    print(f"\n样本分布:")
    print(f"白样本总数: {metrics['white_samples_total']}")
    print(f"黑样本总数: {metrics['black_samples_total']}")
    print(f"总样本数: {metrics['white_samples_total'] + metrics['black_samples_total']}")
    
    print("\n混淆矩阵:")
    cm = metrics['confusion_matrix']
    print(f"       预测白样本   预测黑样本")
    print(f"真实白样本    {cm[1][1]:>8} (TP)    {cm[1][0]:>8} (FN)")
    print(f"真实黑样本    {cm[0][1]:>8} (FP)    {cm[0][0]:>8} (TN)")


def detailed_threshold_analysis(predictions, ground_truth):
    """
    详细分析不同阈值下的precision和recall
    
    Args:
        predictions: 预测结果字典
        ground_truth: 真实标签字典
    """
    # 对齐数据
    aligned_predictions = []
    aligned_truth = []
    
    for sample_id in predictions:
        if sample_id in ground_truth:
            aligned_predictions.append(predictions[sample_id])
            aligned_truth.append(ground_truth[sample_id])
    
    y_true_binary = convert_labels_to_binary(aligned_truth)
    y_scores_white = extract_white_probabilities(aligned_predictions)
    
    # 计算PR曲线
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_scores_white)
    
    print("\n" + "="*80)
    print("详细阈值分析 - Precision和Recall统计")
    print("="*80)
    
    # 选择更多阈值点进行分析
    # detailed_thresholds = [
    #     0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    #     0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.23, 0.26, 0.29,
    #     0.33, 0.37, 0.41, 0.46, 0.51, 0.57, 0.63, 0.70, 0.78, 0.86,
    #     0.92, 0.94, 0.955, 0.97, 0.98, 0.986, 0.992, 0.996, 0.997, 0.998, 0.999
    # ]
    detailed_thresholds = [round(x, 4) for x in np.arange(0.950, 1.000, 0.0005)]
    
    print("阈值\t\t精确率\t\t召回率\t\tF1分数\t\tTP\tFP\tFN\tTN")
    print("-" * 100)
    
    for threshold in detailed_thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores_white]
        cm = confusion_matrix(y_true_binary, y_pred)
        # 注意：confusion_matrix的ravel顺序是 [TN, FP, FN, TP]
        tn, fp, fn, tp = cm.ravel()
        
        # 计算指标
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        print(f"{threshold:.4f}\t\t{p:.4f}\t\t{r:.4f}\t\t{f1:.4f}\t{tp}\t{fp}\t{fn}\t{tn}")
    
    # 分析最佳阈值点
    print("\n" + "="*80)
    print("关键阈值点分析")
    print("="*80)
    
    # 找到不同目标下的最佳阈值
    f1_scores = []
    for i in range(len(precision)):
        if precision[i] + recall[i] > 0:
            f1 = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1 = 0
        f1_scores.append(f1)
    
    # 最佳F1阈值
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
    
    # 高精度阈值（precision > 0.9）
    high_precision_indices = [i for i, p in enumerate(precision) if p > 0.9]
    if high_precision_indices:
        best_high_precision_idx = high_precision_indices[0]
        high_precision_threshold = thresholds[best_high_precision_idx] if best_high_precision_idx < len(thresholds) else thresholds[-1]
    else:
        high_precision_threshold = None
    
    # 高召回阈值（recall > 0.9）
    high_recall_indices = [i for i, r in enumerate(recall) if r > 0.9]
    if high_recall_indices:
        best_high_recall_idx = high_recall_indices[-1]
        high_recall_threshold = thresholds[best_high_recall_idx] if best_high_recall_idx < len(thresholds) else thresholds[-1]
    else:
        high_recall_threshold = None
    
    print(f"最佳F1阈值: {best_f1_threshold:.4f} (F1={f1_scores[best_f1_idx]:.4f})")
    if high_precision_threshold:
        print(f"高精度阈值: {high_precision_threshold:.4f} (Precision>0.9)")
    if high_recall_threshold:
        print(f"高召回阈值: {high_recall_threshold:.4f} (Recall>0.9)")


def evaluate_at_threshold(predictions, ground_truth, threshold, save_path=None, conversations=None):
    """
    在指定的目标阈值下统计准确率/召回情况，并把 FP、FN 样本按 score 得分从大到小排序保存。

    类别约定（与脚本其余部分保持一致）：
        - 白样本=正类(1)，黑样本=负类(0)
        - TP: 真实白 & 预测白
        - FN: 真实白 & 预测黑
        - FP: 真实黑 & 预测白（黑样本被误放过）
        - TN: 真实黑 & 预测黑（黑样本被正确拦截）

    Args:
        predictions: {question_id: score(list 或 数值)}
        ground_truth: {question_id: label}
        threshold: 目标阈值（基于白概率 score[0]）
        save_path: FP/FN 样本导出 JSONL 路径；为 None 时不写文件
        conversations: 可选，{question_id: conversations}，用于在导出的 FP/FN 样本中附带原始对话内容

    Returns:
        dict: 该阈值下的统计指标
    """
    if conversations is None:
        conversations = {}
    # 对齐预测与标签，同时保留 sample_id 与原始 score
    aligned_ids = []
    aligned_predictions = []
    aligned_truth = []
    for sample_id in predictions:
        if sample_id in ground_truth:
            aligned_ids.append(sample_id)
            aligned_predictions.append(predictions[sample_id])
            aligned_truth.append(ground_truth[sample_id])

    if not aligned_ids:
        print("[evaluate_at_threshold] 无对齐样本，跳过")
        return {}

    y_true_binary = convert_labels_to_binary(aligned_truth)
    y_scores_white = extract_white_probabilities(aligned_predictions)
    y_pred_binary = [1 if s >= threshold else 0 for s in y_scores_white]

    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    white_total = sum(y_true_binary)
    black_total = len(y_true_binary) - white_total
    reduction_rate = tp / white_total if white_total > 0 else 0
    false_negative_rate = fn / white_total if white_total > 0 else 0
    # 黑样本误放过率 = FP / 黑样本总数
    fp_rate = fp / black_total if black_total > 0 else 0

    print("\n" + "=" * 60)
    print(f"目标阈值 {threshold:.6f} 下的白样本性能")
    print("=" * 60)
    print(f"白样本精确率 (Precision): {precision:.4f}")
    print(f"白样本召回率 (Recall):    {recall:.4f}")
    print(f"白样本F1分数:             {f1:.4f}")
    print(f"白样本降量率 (TP/白总数):  {reduction_rate:.4f} ({tp}/{white_total})")
    print(f"误杀率 (FN/白总数):        {false_negative_rate:.4f} ({fn}/{white_total})")
    print(f"黑样本误放率 (FP/黑总数):  {fp_rate:.4f} ({fp}/{black_total})")
    print("\n混淆矩阵:")
    print(f"       预测白样本   预测黑样本")
    print(f"真实白样本    {tp:>8} (TP)    {fn:>8} (FN)")
    print(f"真实黑样本    {fp:>8} (FP)    {tn:>8} (TN)")

    # 收集 FP 和 FN 样本，按白概率 score[0] 从大到小排序
    fp_fn_records = []
    for sid, raw_score, true_lbl, pred_lbl, white_p in zip(
            aligned_ids, aligned_predictions, y_true_binary, y_pred_binary, y_scores_white):
        if (true_lbl == 0 and pred_lbl == 1):  # FP: 真实黑样本但被预测为白样本
            category = "FP"
            fp_fn_records.append({
                "question_id": sid,
                "label": ground_truth.get(sid),
                "white_prob": float(white_p),
                "true_label": "black",
                "pred_label": "white",
                "category": category,
                "conversations": conversations.get(sid),
            })
        elif (true_lbl == 1 and pred_lbl == 0):  # FN: 真实白样本但被预测为黑样本
            category = "FN"
            fp_fn_records.append({
                "question_id": sid,
                "label": ground_truth.get(sid),
                "white_prob": float(white_p),
                "true_label": "white",
                "pred_label": "black",
                "category": category,
                "conversations": conversations.get(sid),
            })

    # 按白概率从大到小排序（white_prob 越大越像白样本）
    fp_fn_records.sort(key=lambda x: x["white_prob"], reverse=True)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as fout:
            for rec in fp_fn_records:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nFP/FN 样本（共 {len(fp_fn_records)} 条，FP={fp}, FN={fn}）已按 white_prob 降序保存到: {save_path}")

    return {
        "threshold": threshold,
        "white_precision": precision,
        "white_recall": recall,
        "white_f1": f1,
        "white_reduction_rate": reduction_rate,
        "false_negative_rate": false_negative_rate,
        "fp_rate": fp_rate,
        "confusion_matrix": cm.tolist(),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "white_samples_total": int(white_total),
        "black_samples_total": int(black_total),
        "fp_fn_saved_path": save_path,
        "fp_fn_count": len(fp_fn_records),
    }


def analyze_white_threshold_performance(predictions, ground_truth):
    """
    分析不同阈值下白样本的性能表现
    
    Args:
        predictions: 预测结果字典
        ground_truth: 真实标签字典
    """
    # 对齐数据
    aligned_predictions = []
    aligned_truth = []
    
    for sample_id in predictions:
        if sample_id in ground_truth:
            aligned_predictions.append(predictions[sample_id])
            aligned_truth.append(ground_truth[sample_id])
    
    y_true_binary = convert_labels_to_binary(aligned_truth)
    y_scores_white = extract_white_probabilities(aligned_predictions)
    
    print("\n不同阈值下白样本性能分析:")
    print("-" * 80)
    print("阈值\t\t精确率\t\t召回率\t\tF1分数\t\t降量率")
    print("-" * 80)
    
    # 选择关键阈值进行分析
    key_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98]
    
    for threshold in key_thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores_white]
        cm = confusion_matrix(y_true_binary, y_pred)
        # 注意：confusion_matrix的ravel顺序是 [TN, FP, FN, TP]
        tn, fp, fn, tp = cm.ravel()
        
        # 计算白样本指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        white_total = sum(y_true_binary)
        reduction_rate = tp / white_total if white_total > 0 else 0
        
        print(f"{threshold:.2f}\t\t{precision:.4f}\t\t{recall:.4f}\t\t{f1:.4f}\t\t{reduction_rate:.4f}")
    
    # 调用详细分析
    detailed_threshold_analysis(predictions, ground_truth)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='计算白样本精确率和召回率指标')
    parser.add_argument('--pred_file', type=str, required=True, 
                       help='预测结果文件路径')
    parser.add_argument('--test_file', type=str, required=True,
                       help='测试数据文件路径')
    parser.add_argument('--threshold', type=float, default=None,
                       help='目标阈值（基于白概率 score[0]）；指定后会输出该阈值下的指标，并保存 FP/FN 样本')
    parser.add_argument('--fp_fn_save_path', type=str, default=None,
                       help='FP/FN 样本导出 JSONL 路径；默认保存到 pred_file 同目录的 *_fp_fn_th{threshold}.jsonl')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.pred_file):
        print(f"错误: 预测文件不存在 - {args.pred_file}")
        return
    
    if not os.path.exists(args.test_file):
        print(f"错误: 测试文件不存在 - {args.test_file}")
        return
    
    print(f"预测文件: {args.pred_file}")
    print(f"测试文件: {args.test_file}")
    
    # 提取预测结果和真实标签
    predictions, ground_truth, conversations = extract_predictions_and_labels(args.pred_file, args.test_file)
    
    if not predictions or not ground_truth:
        print("错误: 未能提取到有效的预测结果或真实标签")
        return
    
    print(f"成功加载 {len(predictions)} 个预测结果和 {len(ground_truth)} 个真实标签")
    
    # 计算白样本指标
    metrics = calculate_white_sample_metrics(predictions, ground_truth)
    
    # 打印结果
    print_white_sample_metrics(metrics)
    
    # 分析不同阈值性能
    analyze_white_threshold_performance(predictions, ground_truth)

    # 指定目标阈值时：输出该阈值指标并保存 FP/FN 样本
    if args.threshold is not None:
        if args.fp_fn_save_path:
            fp_fn_save_path = args.fp_fn_save_path
        else:
            pred_dir = os.path.dirname(os.path.abspath(args.pred_file))
            pred_base = os.path.splitext(os.path.basename(args.pred_file))[0]
            fp_fn_save_path = os.path.join(
                pred_dir, f"{pred_base}_fp_fn_th{args.threshold:.4f}.jsonl")
        evaluate_at_threshold(predictions, ground_truth, args.threshold, fp_fn_save_path, conversations)
    
    print("\n" + "="*60)
    print("白样本指标计算完成")
    print("="*60)


if __name__ == '__main__':
    main()