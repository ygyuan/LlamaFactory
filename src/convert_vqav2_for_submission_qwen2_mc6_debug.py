import os
import argparse
import json
from sklearn.metrics import precision_recall_curve
import numpy as np
import torch
import pdb


def get_label_type(label, label_dic):
    """
    将审核标签映射为对应的类型编号并统计标签数量
    Args:
        label: 审核标签
        label_dic: 用于统计各类型标签数量的字典
    Returns:
        int: 标签对应的类型编号(0-6)
    """
    label_map = {
        '100': 0,
        '20746': 1,
        '20007': 2,
        '20012': 3,
        '20002': 4,
    }

    tmp = label_map.get(label, 5)
    label_dic[tmp] = label_dic.get(tmp, 0) + 1

    return tmp


def read_data(args):
    """读取预测结果和测试数据
    
    Args:
        args: 命令行参数对象，包含mejson、ckpt和split属性
        
    Returns:
        tuple: (results, test_split, datalist)
            results: 预测结果字典 {question_id: score}
            test_split: 测试数据列表
            datalist: 测试数据字典 {id: data}
    """
    # 确定输入文件路径
    src = args.mejson if args.mejson else os.path.join(args.ckpt, 'merge.jsonl')
    print("input_file: ", src)
    
    # 读取预测结果
    results = []
    error_lines = []
    try:
        with open(src, 'r') as f:
            for line_idx, line in enumerate(f, 1):
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    error_lines.append(f"Line {line_idx}: {str(e)}")
    except IOError as e:
        print(f"Error opening file {src}: {str(e)}")
        raise

    # 读取测试数据
    test_split = []
    try:
        with open(args.split, 'r') as f:
            # test_split = [json.loads(line) for line in f]
            test_split = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading test split file {args.split}: {str(e)}")
        raise

    # 转换结果格式并创建数据字典
    results = {x['question_id']: x['score'] for x in results}
    datalist = {data['id']: data for data in test_split}

    print(f'total_results: {len(results)}, total_split: {len(test_split)}, error_lines: {len(error_lines)}')
    if error_lines:
        print("First 5 error lines:")
        for err in error_lines[:5]:
            print(err)
            
    return results, test_split, datalist

def process_data(results, test_split):
    """处理测试数据并准备评估指标
    
    Args:
        results: 预测结果字典 {question_id: score}
        test_split: 测试数据列表
        
    Returns:
        tuple: 包含多个评估指标的元组
            all_answers: 提交格式的答案列表
            gt_celue_dic: 黑样本策略分布字典
            gtb_celue_dic: 白样本策略分布字典
            gt_label_dic: 标签类型统计字典
            y_true: 二分类真实标签列表
            y_id: 样本ID列表
            y_audit_label: 原始审核标签列表
            multi_y_scores: 多分类预测分数列表
            y_multi_true: 多分类真实标签列表
    """
    all_answers = []
    
    # 初始化评估指标
    y_true = []
    y_id = []
    y_audit_label = []
    multi_y_scores = []
    y_multi_true = []
    gt_label_dic = {}
    gt_celue_dic = {}
    gtb_celue_dic = {}
    multi_num = {'pic': 0, 'text': 0}

    # 处理每条测试数据
    for x in test_split:
        if x['id'] not in results:
            continue
            
        # 准备提交格式的答案
        all_answers.append({
            'question_id': x['id'],
            'answer': ''
        })

        # 获取预测分数和真实标签
        score = results[x['id']]
        label = x['label']
        
        # 统计图文分布
        pic = 1  # 假设图片数量为1
        multi_num['pic' if pic > 0 else 'text'] += 1

        # 处理策略标签
        celue = x['celue']
        celueid = celue.split("/")[0]  # 获取策略ID

        # 转换二分类标签
        lab = 0 if label == "100" else 1

        # 获取多分类标签并统计
        mlab = get_label_type(label, gt_label_dic)
        
        # 收集评估指标
        y_true.append(lab)
        y_id.append(x['id'])
        y_audit_label.append(label)
        multi_y_scores.append(score)
        y_multi_true.append(mlab)

        # 统计策略分布
        target_dict = gt_celue_dic if label != '100' else gtb_celue_dic
        target_dict[celueid] = target_dict.get(celueid, 0) + 1

    print("图文分布：", multi_num)
    return all_answers, gt_celue_dic, gtb_celue_dic, gt_label_dic, y_true, y_id, y_audit_label, multi_y_scores, y_multi_true


def get_confusion(th_list, mejson, datalist, y_true, y_id, y_audit_label, multi_y_scores, y_multi_true):
    """计算混淆矩阵并输出分类结果
    
    Args:
        th_list: 各分类的阈值列表
        mejson: 输入jsonl文件路径
        datalist: 数据字典
        y_true: 真实标签列表
        y_id: 样本ID列表
        y_audit_label: 审核标签列表
        multi_y_scores: 多分类预测分数列表
        y_multi_true: 多分类真实标签列表
        
    Returns:
        tuple: (confusion, mconfusion, pre_label_dic, pre_celue_dic, preb_celue_dic)
    """
    num_label = 2
    confusion = torch.zeros(num_label, num_label, dtype=torch.long)
    m_label = len(multi_y_scores[0])
    mconfusion = torch.zeros(m_label, m_label, dtype=torch.long)

    # 初始化结果字典
    result_dicts = {
        'pre': {},
        'pre_celue': {},
        'preb_celue': {},
        'corbai': {},
        'errbai': {},
        'louguohei': {},
        'hithei': {}
    }

    # 初始化输出文件列表
    output_files = {
        'louguohei': [],
        'errbai': [],
        'hithei': [],
        'corbai': []
    }

    # 添加表头
    column_names = "\t".join(["id", "score", "pre_label", "tgt_label", "celue", "url", 
                            "audit_label", "pinlun", "beijing"]) + "\n"
    for key in output_files:
        output_files[key].append(column_names)

    try:
        # 处理每个样本
        for i, mscore in enumerate(multi_y_scores):
            id = y_id[i]
            data = datalist[id]
            celue = data['celue']
            celueid = celue.split("/")[0] if "/" in celue else celue
            audit_label = y_audit_label[i]

            # 获取预测结果
            score_tensor = torch.tensor(mscore)
            score, predict_label = torch.max(score_tensor, dim=-1)
            score = float(score)
            pre_mlabel = int(predict_label)

            # 解析评论和背景信息
            # conv = data["conversations"][0]["value"].split("\n")
            conv = data["messages"][0]["content"].split("\n")
            pinlun = conv[-2].split("当前评论：")[1] if len(conv) > 2 else ""
            beijing = conv[-3].split("背景信息：")[1] if len(conv) > 1 else ""
            url = "" # data['image'].replace("/apdcephfs_qy3/share_301069248/data/video", "http://9.22.25.210/xiaoshijie").replace(".jpg", "_0.jpg")

            # 根据阈值判定预测结果
            if (pre_mlabel in (1, 2) and score >= th_list[pre_mlabel]):
                pre_label = 1
                target_dict = 'pre_celue' if y_multi_true[i] != 0 else 'preb_celue'
                label_dict = 'hithei' if y_multi_true[i] != 0 else 'errbai'
            else:
                pre_label = 0
                pre_mlabel = 0
                label_dict = 'louguohei' if y_multi_true[i] != 0 else 'corbai'

            # 更新统计字典
            if y_multi_true[i] != 0 and pre_label == 1:
                result_dicts['pre_celue'][celueid] = result_dicts['pre_celue'].get(celueid, 0) + 1
            elif pre_label == 1:
                result_dicts['preb_celue'][celueid] = result_dicts['preb_celue'].get(celueid, 0) + 1

            # 获取标签类型并生成结果行
            tmp_label = get_label_type(audit_label, result_dicts[label_dict])
            res_line = "\t".join([
                id, f"{score:.6f}", str(pre_mlabel), str(tmp_label),
                celue, url, audit_label, pinlun, beijing
            ]) + "\n"
            output_files[label_dict].append(res_line)

            # 更新混淆矩阵
            confusion[pre_label, y_true[i]] += 1
            mconfusion[pre_mlabel, y_multi_true[i]] += 1

        # 写入结果文件
        for file_type in output_files:
            file_path = mejson.replace(".jsonl", f"_{file_type}.txt")
            with open(file_path, 'w') as f:
                f.writelines(output_files[file_type])

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        raise

    return confusion, mconfusion, result_dicts['pre'], result_dicts['pre_celue'], result_dicts['preb_celue']


def get_confusion_bai(th, mejson, datalist, y_true, y_id, y_audit_label, y_scores_idx, y_true_idx):
    """计算二分类混淆矩阵
    
    Args:
        th: 阈值
        mejson: 输入jsonl文件路径
        datalist: 数据字典
        y_true: 真实标签列表
        y_id: 样本ID列表
        y_audit_label: 审核标签列表
        y_scores_idx: 预测分数列表
        y_true_idx: 真实标签索引列表
        
    Returns:
        tuple: (confusion, pre_label_dic, pre_celue_dic, preb_celue_dic)
    """
    num_label = 2
    confusion = torch.zeros(num_label, num_label, dtype=torch.long)

    # 初始化结果字典和输出列表
    result_dicts = {
        'pre': {},
        'pre_celue': {},
        'preb_celue': {},
        'corbai': {},
        'errbai': {},
        'louguohei': {},
        'hithei': {}
    }
    
    output_lists = {
        'louguohei': [],
        'errbai': [],
        'hithei': [],
        'corbai': []
    }

    # 添加表头
    column_names = "\t".join(["id", "score", "pre_label", "tgt_label", "celue", "url", 
                            "audit_label", "pinlun", "beijing"]) + "\n"
    for key in output_lists:
        output_lists[key].append(column_names)

    try:
        # 处理每个样本
        for i, score in enumerate(y_scores_idx):
            id = y_id[i]
            data = datalist[id]
            celue = data['celue']
            celueid = celue.split("/")[0] if "/" in celue else celue
            audit_label = y_audit_label[i]

            # 解析评论和背景信息
            # conv = data["conversations"][0]["value"].split("\n")
            conv = data["messages"][0]["content"].split("\n")
            # pdb.set_trace()
            pinlun = conv[-2].split("当前评论：")[1] if len(conv) > 2 else ""
            beijing = conv[-3].split("背景信息：")[1] if len(conv) > 1 else ""
            url = "" # data['image'].replace("/apdcephfs_qy3/share_301069248/data/video", "http://9.22.25.210/xiaoshijie").replace(".jpg", "_0.jpg")

            # 根据阈值判定预测结果
            if score >= th:
                pre_label = 1
                get_label_type(audit_label, result_dicts['pre'])
                label_dict = 'hithei' if audit_label != '100' else 'errbai'
                target_dict = 'pre_celue' if audit_label != '100' else 'preb_celue'
            else:
                pre_label = 0
                label_dict = 'louguohei' if audit_label != '100' else 'corbai'

            # 更新策略统计
            if pre_label == 1:
                if audit_label != '100':
                    result_dicts['pre_celue'][celueid] = result_dicts['pre_celue'].get(celueid, 0) + 1
                else:
                    result_dicts['preb_celue'][celueid] = result_dicts['preb_celue'].get(celueid, 0) + 1

            # 生成结果行
            tmp_label = get_label_type(audit_label, result_dicts[label_dict])
            res_line = "\t".join([
                id, f"{score:.6f}", str(pre_label), str(tmp_label),
                celue, url, audit_label, pinlun, beijing
            ]) + "\n"
            output_lists[label_dict].append(res_line)

            # 更新混淆矩阵
            confusion[pre_label, y_true[i]] += 1

        # 写入结果文件
        for file_type in output_lists:
            file_path = mejson.replace(".jsonl", f"_{file_type}_jj.txt")
            with open(file_path, 'w') as f:
                f.writelines(output_lists[file_type])

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        raise

    return confusion, result_dicts['pre'], result_dicts['pre_celue'], result_dicts['preb_celue']

def statistic_celue_results(gt_celue_dic, pre_celue_dic, gtb_celue_dic, preb_celue_dic):
    """统计策略维度的评估结果
    
    Args:
        gt_celue_dic: 黑样本策略分布字典
        pre_celue_dic: 预测黑样本策略分布字典
        gtb_celue_dic: 白样本策略分布字典
        preb_celue_dic: 预测白样本策略分布字典
    """
    # 初始化计数器
    num_gtp0 = num_prep0 = num_gtbp0 = num_prebp0 = 0
    
    # 统计策略维度结果
    for celue_id0 in gt_celue_dic:
        if celue_id0 in pre_celue_dic and celue_id0 in gtb_celue_dic and celue_id0 in preb_celue_dic:
            # 累加计数
            num_gtp0 += gt_celue_dic[celue_id0]
            num_prep0 += pre_celue_dic[celue_id0]
            num_gtbp0 += gtb_celue_dic[celue_id0]
            num_prebp0 += preb_celue_dic[celue_id0]
            
            # 计算各项指标
            total = gt_celue_dic[celue_id0] + gtb_celue_dic[celue_id0]
            gt_ratio = gt_celue_dic[celue_id0] / total
            pre_ratio = pre_celue_dic[celue_id0] / gt_celue_dic[celue_id0]
            preb_ratio = 1 - preb_celue_dic[celue_id0] / gtb_celue_dic[celue_id0]
            
            # 输出策略详细结果
            print(f"{celue_id0}\t{total}\t{gt_ratio:.4f}\t{pre_ratio:.4f}\t{preb_ratio:.4f}\t{pre_celue_dic[celue_id0]}\t{gt_celue_dic[celue_id0]}")

    # 输出汇总结果
    print(f"p0celue hei: {num_prep0/num_gtp0:.4f} {num_prep0} {num_gtp0}")
    print(f"p0celue bai: {num_prebp0/num_gtbp0:.4f} {num_prebp0} {num_gtbp0}")

def get_best_f1(precision, recall, thresholds, topn=5):
    """计算并输出最佳F1值
    
    Args:
        precision: 准确率列表
        recall: 召回率列表
        thresholds: 阈值列表
        topn: 输出前N个最佳结果
    """
    f1_scores = [2*p*r/(p+r+1e-21) for p, r in zip(precision, recall)]
    sorted_results = sorted(zip(precision, recall, f1_scores, thresholds), 
                          key=lambda x: x[2], reverse=True)
    
    for idx, (p, r, f1, th) in enumerate(sorted_results[:topn]):
        print(f"Top {idx+1}: Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}, Threshold={th:.6f}")

def jiangliang_bai():
    """白样本奖励分析"""
    label = 0
    y_true_idx = [int(j != label) for j in y_multi_true]
    y_scores_idx = [1.0 - j[label] for j in multi_y_scores]
    
    print(f"类别 {label} 样本数: {sum(y_true_idx)}")
    precision, recall, thresholds = precision_recall_curve(y_true_idx, y_scores_idx, drop_intermediate=True)
    get_best_f1(precision, recall, thresholds, topn=1)

    # 设置召回率阈值点
    recall_values = [0.9999, 0.999, 0.990, 0.900]
    recall_indices = {value: 0 for value in recall_values}

    # 找到各召回率阈值对应的索引
    for i, v in enumerate(recall):
        for value in recall_values:
            if v >= value:
                recall_indices[value] = i

    # 输出阈值分析结果
    print(f"类别: {label} Recall阈值: {' '.join(map(str, recall_values))}")
    print("对应的Precision:", " ".join(f"{precision[i]:.6f}" for i in recall_indices.values()))
    print("对应的Threshold:", " ".join(f"{thresholds[i]:.6f}" for i in recall_indices.values()))

    # 使用0.99召回率阈值进行分析
    # threshold = thresholds[recall_indices[0.990]]
    threshold = thresholds[recall_indices[args.recall]]
    confusion, pre_label_dic, pre_celue_dic, preb_celue_dic = get_confusion_bai(
        threshold, args.mejson, datalist, y_true, y_id, y_audit_label, y_scores_idx, y_true_idx)
    
    # 输出混淆矩阵和评估指标
    print("混淆矩阵:")
    print(confusion)

    eps = 1e-12
    jiangliang = confusion[0,:].sum().item() / (confusion.sum().item() + eps)
    daji = 1 - jiangliang
    wusha = confusion[1,0].item() / (confusion[:, 0].sum().item() + eps)
    
    print(f"剥白阈值 {threshold:.6f} 降量 {jiangliang:.4f} 打击率 {daji:.4f} 误杀率 {wusha:.4f}")
    print(f'准确率: {confusion[1][1].item() / (confusion[1, :].sum().item() + eps):.4f}')
    print(f'黑样本覆盖率 {confusion[1][1].item() /(confusion[:, 1].sum().item() + eps):.4f}')
    print(f"真实标签分布: {gt_label_dic}, 总数: {sum(gt_label_dic.values())}")

    # 输出各类别评估指标
    for i in range(confusion.size()[0]):
        p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        print(f"标签 {i}: 准确率 {p:.4f}, 召回率 {r:.4f}, F1值 {f1:.4f}")

    # 输出预测标签分布
    print("预测标签分布:", pre_label_dic)
    for index in pre_label_dic:
        try:
            ratio = pre_label_dic[index] / gt_label_dic[index]
            print(f"{index} 召回率 {ratio:.4f} 命中数 {pre_label_dic[index]} 总数 {gt_label_dic[index]}")
        except KeyError:
            print(f"警告: 索引 {index} 不存在于真实标签字典中")

def hit_hei():
    """黑样本命中分析"""
    th_list = []
    num_label = len(gt_label_dic)
    
    # 分析每个黑样本类别
    for label in range(1, num_label):
        y_true_idx = [int(j == label) for j in y_multi_true]
        y_scores_idx = [j[label] for j in multi_y_scores]
        
        print(f"类别 {label} 样本数: {sum(y_true_idx)}")
        precision, recall, thresholds = precision_recall_curve(y_true_idx, y_scores_idx, drop_intermediate=True)
        get_best_f1(precision, recall, thresholds, topn=1)

        # 设置准确率阈值点
        precision_values = [0.990, 0.950, 0.900, 0.500]
        precision_indices = {value: 0 for value in precision_values}
        
        # 找到各准确率阈值对应的索引
        for i, v in enumerate(precision):
            for value in precision_values:
                if v <= value:
                    precision_indices[value] = i

        # 输出阈值分析结果
        print(f"类别: {label} 准确率阈值: {' '.join(map(str, precision_values))}")
        print("对应的覆盖:", " ".join(f"{recall[i]:.6f}" for i in precision_indices.values()))
        print("对应的位置:", " ".join(f"{thresholds[i]:.6f}" for i in precision_indices.values()))
        
        # 使用参数指定的准确率阈值
        th_list.append(thresholds[precision_indices[args.precision]])

    print("阈值列表: ", th_list)
    
    # 使用阈值列表进行混淆矩阵分析
    confusion, mconfusion, pre_label_dic, pre_celue_dic, preb_celue_dic = get_confusion(
        th_list, args.mejson, datalist, y_true, y_id, y_audit_label, multi_y_scores, y_multi_true)
    
    # 输出混淆矩阵
    print("混淆矩阵:")
    print(confusion)
    print("多分类混淆矩阵:")
    print(mconfusion)

    # 计算评估指标
    eps = 1e-12
    jiangliang = confusion[0,:].sum().item() / (confusion.sum().item() + eps)
    daji = 1 - jiangliang
    wusha = confusion[1,0].item() / (confusion[:, 0].sum().item() + eps)
    
    print(f"命中阈值 0.0001 降量 {jiangliang:.4f} 打击率 {daji:.4f} 误杀率 {wusha:.4f}")
    print(f'准确率: {confusion[1][1].item() / (confusion[1, :].sum().item() + eps):.4f}')
    print(f'黑样本覆盖率 {confusion[1][1].item() /(confusion[:, 1].sum().item() + eps):.4f}')
    print(f"真实标签分布: {gt_label_dic}, 总数: {sum(gt_label_dic.values())}")

    # 输出多分类评估指标
    for i in range(mconfusion.size()[0]):
        p = mconfusion[i, i].item() / (mconfusion[i, :].sum().item() + eps)
        r = mconfusion[i, i].item() / (mconfusion[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        print(f"标签 {i}: 准确率 {p:.3f}, 召回率 {r:.3f}, F1值 {f1:.3f}")

    # 输出预测标签分布
    for i in range(11):
        if i in pre_label_dic:
            print(f"{i} 预测数 {pre_label_dic[i]} 真实数 {gt_label_dic[i]}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--mejson', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--recall', type=float, default=0.999)
    parser.add_argument('--precision', type=float, default=0.900)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    results, test_split, datalist = read_data(args)
    all_answers, gt_celue_dic, gtb_celue_dic, gt_label_dic, y_true, y_id, y_audit_label, multi_y_scores, y_multi_true = process_data(results, test_split)
    jiangliang_bai()
    hit_hei()
