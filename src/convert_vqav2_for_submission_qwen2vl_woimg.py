import os
import argparse
import json
from sklearn.metrics import precision_recall_curve
import sys
import torch
import re
import numpy as np
from tqdm import tqdm
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
        '20430': 0, '20429': 0, '20001': 0, '20487': 0, '20457': 0, '20431': 0, '20202': 0,
        '20470': 1, '21022': 1, '20004': 1,
        '20002': 2, '20103': 2, '20012': 2, '20656': 2,
        '20006': 3, '20746': 3,
        '21016': 4,
        '100': 6
    }

    tmp = label_map.get(label, 5)
    label_dic[tmp] = label_dic.get(tmp, 0) + 1

    return tmp


def get_best_f1(precision, recall, thresholds):
    f1 = [] 
    for idx, th in enumerate(thresholds):
        f1.append(2*precision[idx]*recall[idx]/(precision[idx]+recall[idx]+1e-21))
    f1 = np.array(f1)
     # 将precision, recall, F1值和thresholds组合在一起
    data = list(zip(precision, recall, f1, thresholds))
    # 按照F1值由大到小排序
    sorted_data = sorted(data, key=lambda x: x[2], reverse=True)
    # pdb.set_trace()
    topn=5  
    for idx, item in enumerate(sorted_data[:topn]):
        print(idx, item)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="./eval/vqav2")
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--mejson', type=str, default="")
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--precision', type=float, default=0.900)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.mejson!="":
        src= args.mejson
    else:
        src=  os.path.join( args.ckpt, 'merge.jsonl')
    print(src)
    dst =  os.path.join( args.ckpt, 'res.jsonl')


    results = []
    error_line = 0
    for line_idx, line in enumerate(open(src)):
        try:
            results.append(json.loads(line))
        except:
            error_line += 1
    results = {x['question_id']: x['score'] for x in results}

    with open(args.split, 'r') as f:
        test_split = json.load(f)
    split_ids = set()
    datalist = {}
    for x in test_split:
        ids = x['id']
        split_ids.add(ids)
        datalist[ids] = x
    print(f'total results: {len(results)}, total split: {len(test_split)}, error_line: {error_line}')

    all_answers = []
    y_scores = []
    y_id=[]
    gt_label_dic = {}
    y_true=[]
    y_audit_label=[]
    gt_celue_dic = {}
    gtb_celue_dic = {}

    cnt = 0    
    jl_cnt=0
    right_cnt = 0
    wrong_cnt = 0 
    pre_label_dic = {}
    pre_celue_dic = {}

    preb_celue_dic = {} 
    err_label_dict={}
    err_dict=[]

    num_label=2

    errinfo_dict=[]
    errbaiinfo_dict=[]
    for i in range(6):
        d=[]
        errinfo_dict.append(d)

    err99_file = open("err99.txt", 'w')
    errbai99_file=open("errbai99.txt", 'w')
    
    multi_num={'pic':0,'text':0}


    for x in test_split:
        if x['id'] in results:
            all_answers.append({
                'question_id': x['id'],
                'answer': ''
            })

            score = results[x['id']]
            label = x['label']
            pic=int(x['picnum'])
            if pic>0:
               multi_num['pic']+=1
            else:
               multi_num['text']+=1   
            if "onlinescore" in x:
                onlinescore=float(x['onlinescore'])
                # pic=int(x['picnum'])
                if pic==0:
                    if onlinescore < 0.013:
                        jl_cnt=jl_cnt+1
                else:
                    if onlinescore < 0.0073:
                        jl_cnt=jl_cnt+1
                if onlinescore > 0.96:
                    jl_cnt=jl_cnt+1


            celue=x['celue']
            p=celue.find("/")
            if p!=-1:
                celueid=celue[:p]
            else:
                celueid=celue

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

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    get_best_f1(precision, recall, thresholds)
    # pdb.set_trace()

    # 找到不同召回率对应的阈值位置
    recall_values = [0.995, 0.990, 0.980, 0.950, 0.900]
    recall_indices = {value: 0 for value in recall_values}

    for i, v in enumerate(recall):
        for value in recall_values:
            if v >= value:
                 recall_indices[value] = i

    print("recall阈值:", " ".join(map(str, recall_values)))
    print("对应的位置:", " ".join("{:.6f}".format(thresholds[i]) for i in recall_indices.values()))


    # 找到最接近0.99阈值的下标
    precision_values = [0.995, 0.990, 0.980, 0.950, 0.900]
    precision_indices = {value: 0 for value in precision_values}
    for i, v in enumerate(precision):
        for value in precision_values:
            if v <= value:
                precision_indices[value] = i
    print("precision阈值: ", " ".join(map(str, recall_values)))
    print("对应的位置:", " ".join("{:.6f}".format(thresholds[i]) for i in precision_indices.values()))
    # pdb.set_trace()

    print("图文分布：",multi_num)
    confusion = torch.zeros(num_label, num_label, dtype=torch.long)

    # th = thresholds[recall_indices[args.precision]]
    th = thresholds[precision_indices[args.precision]]
    for i, score in enumerate(y_scores):
        id=y_id[i]
        celue=datalist[id]['celue']
        p=celue.find("/")
        if p!=-1:
            celueid=celue[:p]
        else:
            celueid=celue
        audit_label = y_audit_label[i]

        qs = datalist[id]["messages"][0]["content"].split("\n")
        pinlun = qs[2].split("当前评论：")[1]
        beijing = qs[1].split("背景信息：")[1]
        url = ""

        if score >= th:
            pre_label=1
            get_label_type(audit_label, pre_label_dic)

            if audit_label!='100':
                if celueid not in pre_celue_dic:
                    pre_celue_dic[celueid] = 1
                else:
                    pre_celue_dic[celueid] += 1
            else:
                if celueid not in preb_celue_dic:
                    preb_celue_dic[celueid] = 1
                else:
                    preb_celue_dic[celueid] += 1
            
                tmp_label=get_label_type(audit_label, err_label_dict)
                celue=datalist[id]['celue']
                res=id+"\t"+str(score)+"\t"+audit_label+"\t"+celue+"\t"+pinlun+"\t"+beijing+"\t"+str(pic)+"\n"
                errbaiinfo_dict.append(res)
        
                
        else:
            pre_label=0
            if audit_label!='100':
                tmp_label=get_label_type(audit_label, err_label_dict)
                celue=datalist[id]['celue']
                res=id+"\t"+str(score)+"\t"+audit_label+"\t"+celue+"\t"+pinlun+"\t"+beijing+"\t"+str(pic)+"\n"
                errinfo_dict[tmp_label].append(res)
        # celue=datalist[id]['celue']
        confusion[pre_label,y_true[i]]+=1


    for i in range(6):
        for j, path in enumerate(errinfo_dict[i]):
            err99_file.writelines(path)
    errbai99_file.writelines(errbaiinfo_dict)

    print("Confusion matrix:")
    print(confusion)
    t_sum = torch.sum(confusion).data
    hei_sum = (confusion[0][1]+confusion[1][1]).data
    bai_sum = (confusion[0][0]+confusion[1][0]).data
    print(th, t_sum, bai_sum,hei_sum)
    print('指黑整体: ', (confusion[1][0].data+confusion[1][1].data)/t_sum, " 整体打击: ", 1 - (confusion[1][0].data+confusion[1][1].data)/t_sum )
    print('白样本误杀: ',confusion[1][0].data/bai_sum)
    print('准确率: ', confusion[1][1].data / (confusion[1][1].data + confusion[1][0].data))
    print('黑样本覆盖率', confusion[1][1].data/hei_sum)
    print("Report precision, recall, and f1:")
    eps = 1e-9
    for i in range(confusion.size()[0]):
        p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    print(pre_label_dic)
    for i in list(pre_label_dic.keys()):
        print(i, pre_label_dic[i]/gt_label_dic[i], pre_label_dic[i], gt_label_dic[i])
    #print("政治: ", pre_label_dic[0]/gt_label_dic[0], pre_label_dic[0], gt_label_dic[0])
    #print("黑产: ", pre_label_dic[1]/gt_label_dic[1], pre_label_dic[1], gt_label_dic[1])
    #print("色情: ", pre_label_dic[2]/gt_label_dic[2], pre_label_dic[2], gt_label_dic[2])
    # print(3, pre_label_dic[3]/gt_label_dic[3], pre_label_dic[3], gt_label_dic[3])
    # print(4, pre_label_dic[4]/gt_label_dic[4], pre_label_dic[4], gt_label_dic[4])
    # print(5, pre_label_dic[5]/gt_label_dic[5], pre_label_dic[5], gt_label_dic[5])
    #print(6, pre_label_dic[6]/gt_label_dic[6], pre_label_dic[6], gt_label_dic[6])


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
            #print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(celue_id0,gt_celue_dic[celue_id0]+gtb_celue_dic[celue_id0],gt_celue_dic[celue_id0]/(gt_celue_dic[celue_id0]+gtb_celue_dic[celue_id0]),\
            #                pre_celue_dic[celue_id0]/gt_celue_dic[celue_id0],1-preb_celue_dic[celue_id0]/gtb_celue_dic[celue_id0],pre_celue_dic[celue_id0],gt_celue_dic[celue_id0]))        

    print("p0celue: ",num_prep0/(num_gtp0+eps),num_prep0,num_gtp0)
    print("p0celue bai: ",num_prebp0/(num_gtbp0+eps),num_prebp0,num_gtbp0)
    with open(dst, 'w') as f:
        json.dump(all_answers, open(dst, 'w'))
