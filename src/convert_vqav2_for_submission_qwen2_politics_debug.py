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
        '20430': 1, '20429': 1, '20001': 1, '20457': 1, '20431': 1, '20202': 1,
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

    # 读取预测结果
    results = []
    error_line = 0
    for line_idx, line in enumerate(open(src)):
        try:
            results.append(json.loads(line))
        except:
            error_line += 1

    results = {x['question_id']: x['score'] for x in results}
    test_split = [json.loads(line) for line in open(args.split)]
    #with open(args.split, 'r') as f:
    #    test_split = json.load(f)

    # 读取测试数据
    datalist = {}
    for data in test_split:
        id=data['id']
        datalist[id]=data

    print(f'total_results: {len(results)}, total_split: {len(test_split)}, error_line: {error_line}')
    return results, test_split, datalist

def process_data(results, test_split):
    all_answers = []

    # 初始化评估所需的变量
    y_scores = []
    y_id=[]
    gt_label_dic = {}
    y_true=[]
    y_audit_label=[]
    gt_celue_dic = {}
    gtb_celue_dic = {}
    multi_y_scores = []
    y_multi_true = []

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
            # 处理策略标签
            celue = x['celue'] #="部分自动打击词库转送审"
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

            mlab = get_label_type(label, gt_label_dic)
            y_true.append(lab) 
            y_id.append(x['id'])
            y_audit_label.append(label)
            multi_y_scores.append(score)
            y_multi_true.append(mlab)

            # 统计策略分布
            if label != '100':
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
    return all_answers, gt_celue_dic, gtb_celue_dic, gt_label_dic, y_true, y_id, y_audit_label, multi_y_scores, y_multi_true


def get_confusion(th_list, mejson, datalist, y_true, y_id, y_audit_label, multi_y_scores, y_multi_true):
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

    louguohei_file = open(mejson.replace(".jsonl", "")+"_louguohei.txt", 'w')
    errbai_file=open(mejson.replace(".jsonl", "")+"_errbai.txt", 'w')
    hithei_file = open(mejson.replace(".jsonl", "")+"_hithei.txt", 'w')
    corbai_file = open(mejson.replace(".jsonl", "")+"_corbai.txt", 'w')

    m_label = len(multi_y_scores[0])
    mconfusion = torch.zeros(m_label, m_label, dtype=torch.long)
    #tmp_idx = recall_indices[args.precision]
    #tmp_idx = precision_indices[precision]
    #th=thresholds[tmp_idx]
    #th = 0.0001
    for i, mscore in enumerate(multi_y_scores):
        id = y_id[i]
        celue = datalist[id]['celue']
        p = celue.find("/")
        if p != -1:
            celueid = celue[:p]
        else:
            celueid = celue
        audit_label = y_audit_label[i]

        scoretensor = torch.tensor(mscore)
        score, predict_label = torch.max(scoretensor, dim=-1)
        score= float(score)
        pre_mlabel = int(predict_label)

        #qs = datalist[id]["conversations"][0]["value"].split("\n")
        #pinlun = qs[2].split("当前评论：")[1]
        #beijing = qs[1].split("背景信息：")[1]
        # url = datalist[id]['image'].replace("/apdcephfs_qy3/share_301069248/data/video","http://9.22.25.210/xiaoshijie")

        qs = datalist[id]["input"].split("\n")
        pinlun = qs[1].split("当前评论：")[1]
        beijing = qs[0].split("背景信息：")[1]
        url = ""

        #qs = datalist[id]["messages"][0]["content"].split("\n")
        #pinlun = qs[2].split("当前评论：")[1]
        #beijing = qs[1].split("背景信息：")[1]
        #url = datalist[id]['images'][0].replace("/apdcephfs_qy3/share_301069248/data/video","http://9.22.25.210/xiaoshijie")

        # 根据阈值判定预测结果
        #if (pre_mlabel != 0):
        if (pre_mlabel == 1 and score >= th_list[pre_mlabel]) or \
           (pre_mlabel == 2 and score >= th_list[pre_mlabel]):
            pre_label=1
            # if pre_mlabel == y_multi_true[i]:
            if y_multi_true[i] != 0:
                if celueid not in pre_celue_dic:
                    pre_celue_dic[celueid] = 1
                else:
                    pre_celue_dic[celueid] += 1
                tmp_label=get_label_type(audit_label, hithei_label_dict)
                res = "\t".join([id, "{:.6f}".format(score), str(pre_mlabel), str(tmp_label), celue, url, audit_label, pinlun, beijing])+"\n"
                hithei_list.append(res)
            else:
                if celueid not in preb_celue_dic:
                    preb_celue_dic[celueid] = 1
                else:
                    preb_celue_dic[celueid] += 1
                tmp_label=get_label_type(audit_label, errbai_label_dict)
                res = "\t".join([id, "{:.6f}".format(score), str(pre_mlabel), str(tmp_label), celue, url, audit_label, pinlun, beijing])+"\n"
                errbai_list.append(res)
        else:
            pre_label = 0
            pre_mlabel = 0
            if y_multi_true[i] != 0:
                tmp_label=get_label_type(audit_label, louguohei_label_dict)
                res = "\t".join([id, "{:.6f}".format(score), str(pre_mlabel), str(tmp_label), celue, url, audit_label, pinlun, beijing])+"\n"
                louguohei_list.append(res)
            else:
                tmp_label=get_label_type(audit_label, corbai_label_dict)
                res = "\t".join([id, "{:.6f}".format(score), str(pre_mlabel), str(tmp_label), celue, url, audit_label, pinlun, beijing])+"\n"
                corbai_list.append(res)
        confusion[pre_label, y_true[i]]+=1
        mconfusion[pre_mlabel,y_multi_true[i]]+=1

    # 写入错误案例
    louguohei_file.writelines(louguohei_list)
    corbai_file.writelines(corbai_list)
    errbai_file.writelines(errbai_list)
    hithei_file.writelines(hithei_list)

    print("hit_hei: ", hithei_label_dict)
    print("err_bai: ", errbai_label_dict)
    print("louguo_hei: ", louguohei_label_dict)
    print("cor_bai: ", corbai_label_dict)
    return confusion, mconfusion, pre_celue_dic, preb_celue_dic 


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
        print(item)

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
    all_answers, gt_celue_dic, gtb_celue_dic, gt_label_dic, y_true, y_id, y_audit_label, multi_y_scores, y_multi_true = process_data(results, test_split)

    th_list = []
    num_label = len(gt_label_dic)
    for label in range(num_label):
        y_true_idx = [ int(j==label) for j in y_multi_true]
        y_scores_idx = [j[label] for j in multi_y_scores]
        print(label, sum(y_true_idx))
        precision, recall, thresholds = precision_recall_curve(y_true_idx, y_scores_idx, drop_intermediate=True)
        get_best_f1(precision, recall, thresholds, topn=1)

        # 找到最接近0.99阈值的下标0
        precision_values = [0.999, 0.950, 0.900, 0.700, 0.500]
        precision_indices = {value: 0 for value in precision_values}
        for i, v in enumerate(precision):
            for value in precision_values:
                if v <= value:
                    precision_indices[value] = i
        print("类别: ",  label, " precision阈值: ", " ".join(map(str, precision_values)))
        print("对应的覆盖:", " ".join("{:.6f}".format(recall[i]) for i in precision_indices.values()))
        print("对应的位置:", " ".join("{:.6f}".format(thresholds[i]) for i in precision_indices.values()))
        th_list.append(thresholds[precision_indices[args.precision]])
    # pdb.set_trace()

    print("th_list: ", th_list)
    print("gt_label_dic: ", gt_label_dic)
    th = 0.0001
    confusion, mconfusion, pre_celue_dic, preb_celue_dic = get_confusion(th_list, args.mejson, datalist, y_true, y_id, y_audit_label, multi_y_scores, y_multi_true)
    print("Confusion matrix:")
    print(confusion)
    print(mconfusion)

    eps = 1e-12
    jiangliang = confusion[0,:].sum().item() / (confusion.sum().item() + eps)
    daji = 1 - jiangliang

    wusha = confusion[1,0].item() / (confusion[: , 0].sum().item() + eps)
    print("threshold {:.6f} jiangliang {:.4f} daji {:.4f} wusha {:.4f}".format(th, jiangliang, daji, wusha))
    print('准确率: ', confusion[1][1].item() / (confusion[1, :].sum().item() + eps) )
    print('黑样本覆盖率', confusion[1][1].item() /(confusion[: , 1].sum().item() + eps) )
    print("gt_label_dic: ", gt_label_dic, sum(gt_label_dic.values()))

    for i in range(mconfusion.size()[0]):
        p = mconfusion[i, i].item() / (mconfusion[i, :].sum().item() + eps)
        r = mconfusion[i, i].item() / (mconfusion[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        print("Label {:d},{:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
        # print("Label {}: {:.3f},{:.3f}, {:.3f}, {:.3f}".format(i, p, p2, r, f1))

    # statistic_celue_results(gt_celue_dic, pre_celue_dic, gtb_celue_dic, preb_celue_dic)
