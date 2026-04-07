import os
import argparse
import json
from sklearn.metrics import precision_recall_curve
import sys
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
        '20430': 0, '20429': 0, '20001': 0, '20457': 0, '20431': 0, '20202': 0,
        '20004': 1,
        '100': 3
    }

    tmp = label_map.get(label, 2)
    label_dic[tmp] = label_dic.get(tmp, 0) + 1

    return tmp


def read_data(args):
    # 确定输入文件路径
    if args.mejson!="":
        src= args.mejson
    else:
        src=  os.path.join( args.ckpt, 'merge.jsonl')
    print("input_file: ", src)
    test_split=args.split

    # 读取预测结果
    results = []
    error_line = 0
    for line_idx, line in enumerate(open(src)):
        try:
            results.append(json.loads(line))
        except:
            error_line += 1

    results = {x['question_id']: x['score'] for x in results}
    test_split = [json.loads(line) for line in open(test_split)]
    split_ids = set([x['id'] for x in test_split])

    # 读取测试数据
    datalist = {}
    with open(args.split, "r") as f:
        lines = f.readlines()
        for line in lines:
            data=json.loads(line.strip())
            id=data['id']
            datalist[id]=data

    print(f'total_results: {len(results)}, total_split: {len(test_split)}, error_line: {error_line}')
    return results, test_split, datalist

def process_data(args):
    all_answers = []

    # 初始化评估所需的变量
    y_scores = []
    y_id=[]
    gt_label_dic = {}
    y_true=[]
    y_audit_label=[]
    gt_celue_dic = {}
    gtb_celue_dic = {}

    jl_cnt=0
    multi_num={'pic':0,'text':0}

    # 处理每条测试数据
    for x in test_split:
        if x['id'] in results:
            all_answers.append({
                'question_id': x['id'],
                'answer': ''
            })

            score = results[x['id']]
            label = x['label']
            pic=1 #int(x['picnum'])
            if pic>0:
               multi_num['pic']+=1
            else:
               multi_num['text']+=1   

            # 处理在线分数
            if "onlinescore" in x:
                onlinescore=float(x['onlinescore'])
                if pic==0:
                    if onlinescore < 0.013:
                        jl_cnt=jl_cnt+1
                else:
                    if onlinescore < 0.0073:
                        jl_cnt=jl_cnt+1
                if onlinescore > 0.96:
                    jl_cnt=jl_cnt+1

            # 处理策略标签
            celue=x['celue']="部分自动打击词库转送审"
            p=celue.find("/")
            if p!=-1:
                celueid=celue[:p]
            else:
                celueid=celue

            # 转换标签
            if label=="100":
                lab=0
            else:
                lab=1

            audit_label=label
            y_true.append(lab)
            y_scores.append(score)
            y_audit_label.append(label)
            get_label_type(label, gt_label_dic)

            y_id.append(x['id'])

            # 统计策略分布
            if audit_label!='100':
                if celueid not in gt_celue_dic:
                    gt_celue_dic[celueid] = 1
                else:
                    gt_celue_dic[celueid] += 1
            else:
                if celueid not in gtb_celue_dic:
                    gtb_celue_dic[celueid] = 1
                else:
                    gtb_celue_dic[celueid] += 1
    print("图文分布：",multi_num)
    return all_answers, gt_celue_dic, gtb_celue_dic, gt_label_dic, y_true, y_scores, y_id, y_audit_label

def statistic_celue_results(gt_celue_dic, pre_celue_dic, gtb_celue_dic, preb_celue_dic):
    # 统计策略维度的评估结果
    num_gtp0=0
    num_prep0=0
    num_gtbp0=0
    num_prebp0=0
    for celue_id0 in gt_celue_dic :
        if celue_id0 in gt_celue_dic and celue_id0 in pre_celue_dic and celue_id0 in gtb_celue_dic and celue_id0 in preb_celue_dic:
            num_gtp0=num_gtp0+gt_celue_dic[celue_id0]
            num_prep0=num_prep0+pre_celue_dic[celue_id0]
            num_gtbp0=num_gtbp0+gtb_celue_dic[celue_id0]
            num_prebp0=num_prebp0+preb_celue_dic[celue_id0]
            re=pre_celue_dic[celue_id0]/gt_celue_dic[celue_id0]
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(celue_id0,gt_celue_dic[celue_id0]+gtb_celue_dic[celue_id0],gt_celue_dic[celue_id0]/(gt_celue_dic[celue_id0]+gtb_celue_dic[celue_id0]),\
                            pre_celue_dic[celue_id0]/gt_celue_dic[celue_id0],1-preb_celue_dic[celue_id0]/gtb_celue_dic[celue_id0],pre_celue_dic[celue_id0],gt_celue_dic[celue_id0]))        

    print("p0celue hei: ",num_prep0/num_gtp0,num_prep0,num_gtp0)
    print("p0celue bai: ",num_prebp0/num_gtbp0,num_prebp0,num_gtbp0)

    # 输出策略维度的详细结果
    for celue_id0 in gt_celue_dic :
        if celue_id0 in gt_celue_dic and celue_id0 in pre_celue_dic and celue_id0 in gtb_celue_dic and celue_id0 in preb_celue_dic:
            num_gtp0=num_gtp0+gt_celue_dic[celue_id0]
            num_prep0=num_prep0+pre_celue_dic[celue_id0]
            num_gtbp0=num_gtbp0+gtb_celue_dic[celue_id0]
            num_prebp0=num_prebp0+preb_celue_dic[celue_id0]
            re=pre_celue_dic[celue_id0]/gt_celue_dic[celue_id0]
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(celue_id0,gt_celue_dic[celue_id0]+gtb_celue_dic[celue_id0],gt_celue_dic[celue_id0]/(gt_celue_dic[celue_id0]+gtb_celue_dic[celue_id0]),\
                            pre_celue_dic[celue_id0]/gt_celue_dic[celue_id0],preb_celue_dic[celue_id0]/gtb_celue_dic[celue_id0],pre_celue_dic[celue_id0],gt_celue_dic[celue_id0]))    


def get_best_f1(precision, recall, thresholds, topn=5):
    f1 = []
    for idx, th in enumerate(thresholds):
        f1.append(2*precision[idx]*recall[idx]/(precision[idx]+recall[idx]+1e-21))
    f1 = np.array(f1)
     # 将precision, recall, F1值和thresholds组合在一起
    data = list(zip(precision, recall, f1, thresholds))
    # 按照F1值由大到小排序
    sorted_data = sorted(data, key=lambda x: x[2], reverse=True)
    # pdb.set_trace()
    # topn=5
    for idx, item in enumerate(sorted_data[:topn]):
        print(idx, item)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--mejson', type=str, default="")
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--precision', type=float, default=0.900)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    results, test_split, datalist = read_data(args)
    all_answers, gt_celue_dic, gtb_celue_dic, gt_label_dic, y_true, y_scores, y_id, y_audit_label = process_data(args)

    # 计算PR曲线
    # pdb.set_trace()
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    get_best_f1(precision, recall, thresholds, topn=1)

    # 找到不同召回率对应的阈值位置
    recall_values = [0.990, 0.980, 0.950, 0.900]
    recall_indices = {value: 0 for value in recall_values}

    for i, v in enumerate(recall):
        for value in recall_values:
            if v >= value:
                 recall_indices[value] = i

    print("recall阈值:", " ".join(map(str, recall_values)))
    print("对应的位置:", " ".join("{:.6f}".format(thresholds[i]) for i in recall_indices.values()))


    # 找到最接近0.99阈值的下标0
    precision_values = [0.990, 0.980, 0.950, 0.900]
    precision_indices = {value: 0 for value in precision_values}
    for i, v in enumerate(precision):
        for value in precision_values:
            if v <= value:
                precision_indices[value] = i
    print("precision阈值: ", " ".join(map(str, recall_values)))
    print("对应的位置:", " ".join("{:.6f}".format(thresholds[i]) for i in precision_indices.values()))

    num_label=2
    # 计算混淆矩阵
    confusion = torch.zeros(num_label, num_label, dtype=torch.long)


    pre_label_dic = {}
    pre_celue_dic = {}
    preb_celue_dic = {}

    corbai_label_dict = {}
    errbai_label_dict = {}
    louguohei_label_dict = {}
    hithei_label_dict={}

    louguohei_list=[]
    errbai_list=[]
    hithei_list = []
    corbai_list = []

    louguohei_file = open(args.mejson.replace(".jsonl", "")+"_louguohei.txt", 'w')
    errbai_file=open(args.mejson.replace(".jsonl", "")+"_errbai.txt", 'w')
    hithei_file = open(args.mejson.replace(".jsonl", "")+"_hithei.txt", 'w')
    corbai_file = open(args.mejson.replace(".jsonl", "")+"_corbai.txt", 'w')
    # tmp_idx = recall_indices[0.990]
    # tmp_idx = precision_indices[0.900]
    #tmp_idx = recall_indices[0.990]
    tmp_idx = precision_indices[args.precision]

    # th=thresholds[tmp_idx]
    th = 0.4195 
    # 根据阈值进行预测并统计结果
    for i, score in enumerate(y_scores):
        id=y_id[i]
        celue = datalist[id]['celue'] # = "部分自动打击词库转送审"
        p=celue.find("/")
        if p!=-1:
            celueid=celue[:p]
        else:
            celueid=celue
        audit_label = y_audit_label[i]

        pic=1 #int(datalist[id]['picnum'])

        qs = datalist[id]["input"].split("<sep>")
        pinlun = qs[0].split("当前评论：")[1]
        beijing = qs[1].split("背景信息：")[1]

        #qs = datalist[id]["input"].split("\n")
        #pinlun = qs[1].split("当前评论：")[1]
        #beijing = qs[0].split("背景信息：")[1]
        url = "" #datalist[id]['image'].replace("/apdcephfs_qy3/share_301069248/data/video","http://9.22.25.210/xiaoshijie")
        
        # 根据阈值判定预测结果
        if score >= th:
            pre_label=1
            # pdb.set_trace()
            get_label_type(audit_label, pre_label_dic)
            # 统计策略分布
            if audit_label!='100':
                if celueid not in pre_celue_dic:
                    pre_celue_dic[celueid] = 1
                else:
                    pre_celue_dic[celueid] += 1
                tmp_label=get_label_type(audit_label, hithei_label_dict)
                res = "\t".join([id, str(score), str(pre_label), str(tmp_label), celue, url, audit_label, pinlun, beijing])+"\n"
                hithei_list.append(res)
            else:
                if celueid not in preb_celue_dic:
                    preb_celue_dic[celueid] = 1
                else:
                    preb_celue_dic[celueid] += 1
            
                tmp_label=get_label_type(audit_label, errbai_label_dict)
                res = "\t".join([id, str(score), str(pre_label), str(tmp_label), celue, url, audit_label, pinlun, beijing])+"\n"
                errbai_list.append(res)
        else:
            pre_label=0
            if audit_label!='100':
                tmp_label=get_label_type(audit_label, louguohei_label_dict)
                res = "\t".join([id, str(score), str(pre_label), str(tmp_label), celue, url, audit_label, pinlun, beijing])+"\n"
                louguohei_list.append(res)
            else:
                tmp_label=get_label_type(audit_label, corbai_label_dict)
                res = "\t".join([id, str(score), str(pre_label), str(tmp_label), celue, url, audit_label, pinlun, beijing])+"\n"
                corbai_list.append(res)
        confusion[pre_label,y_true[i]]+=1

    # 写入错误案例
    louguohei_file.writelines(louguohei_list)
    corbai_file.writelines(corbai_list)
    errbai_file.writelines(errbai_list)
    hithei_file.writelines(hithei_list)

    print("Confusion matrix:")
    print(confusion)
    eps = 1e-12
    jiangliang = confusion[0,:].sum().item() / (confusion.sum().item() + eps)
    daji = 1 - jiangliang
    # wusha =  confusion[1,0].item() / (confusion[1, :].sum().item() + eps)
    wusha = confusion[1,0].item() / (confusion[: , 0].sum().item() + eps)
    print("threshold {:.6f} jiangliang {:.4f} daji {:.4f} wusha {:.4f}".format(th, jiangliang, daji, wusha))
    print('准确率: ', confusion[1][1].item() / (confusion[1, :].sum().item() + eps) )
    print('黑样本覆盖率', confusion[1][1].item() /(confusion[: , 1].sum().item() + eps) )
    for i in range(confusion.size()[0]):
        p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        print("Label {}: precision {:.4f}, recall {:.4f}, f1 {:.4f}".format(i, p, r, f1))
    
    # 输出各类型标签的预测结果
    print("pre_label_dic: ", pre_label_dic)
    print("gt_label_dic: ", gt_label_dic)
    
    # 定义需要输出的标签类型索引
    #label_indices = [0, 1, 2, 3, 6]
    label_indices = sorted(list(pre_label_dic.keys()))
    for index in label_indices:
        try:
            ratio = pre_label_dic[index] / gt_label_dic[index]
            print("{} recall {:.4f} hit {} all {}".format(index, ratio, pre_label_dic[index], gt_label_dic[index]))
        except KeyError as e:
            print(f"索引 {index} 不存在于标签字典中: {e}")

    # statistic_celue_results(gt_celue_dic, pre_celue_dic, gtb_celue_dic, preb_celue_dic)
