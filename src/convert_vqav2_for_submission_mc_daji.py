import os
import argparse
import json
from sklearn.metrics import precision_recall_curve
import numpy as np
import torch
import pdb


def get_label_type(label, label_dic={}):
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
        '20001': 1, 
        '20002': 1, 
        '20004': 1, 
        '20006': 1, 
        '20007': 1, 
        '20012': 1, 
        '20202': 1, 
        '20429': 1, 
        '20430': 1, 
        '20431': 1, 
        '20746': 1, 
        '21016': 1, 
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
    for line in open(src):
        try:
            results.append(json.loads(line))
        except:
            error_line += 1

    results = {x['question_id']: x['score'] for x in results}
    test_split = [json.loads(line) for line in open(args.split)]

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

def statistic_celue_results(gt_celue_dic, pre_celue_dic, gtb_celue_dic, preb_celue_dic):
    num_gtp0 = 0
    num_prep0 = 0
    num_gtbp0 = 0
    num_prebp0 = 0
    for celue_id0 in gt_celue_dic:
        if celue_id0 not in preb_celue_dic:
            preb_celue_dic[celue_id0]=0
        if celue_id0 not in pre_celue_dic:
            pre_celue_dic[celue_id0]=0
        if celue_id0 in gt_celue_dic and celue_id0 in pre_celue_dic and celue_id0 in gtb_celue_dic and celue_id0 in preb_celue_dic:
            num_gtp0 = num_gtp0+gt_celue_dic[celue_id0]
            num_prep0 = num_prep0+pre_celue_dic[celue_id0]
            num_gtbp0 = num_gtbp0+gtb_celue_dic[celue_id0]
            num_prebp0 = num_prebp0+preb_celue_dic[celue_id0]
            re = pre_celue_dic[celue_id0]/gt_celue_dic[celue_id0]
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(celue_id0, gt_celue_dic[celue_id0]+gtb_celue_dic[celue_id0], gt_celue_dic[celue_id0]/(gt_celue_dic[celue_id0]+gtb_celue_dic[celue_id0]),
                                                      pre_celue_dic[celue_id0]/gt_celue_dic[celue_id0], 1-preb_celue_dic[celue_id0]/gtb_celue_dic[celue_id0], pre_celue_dic[celue_id0], gt_celue_dic[celue_id0]))

    print("p0celue: ", num_prep0/num_gtp0, num_prep0, num_gtp0)
    print("p0celue bai: ", num_prebp0/num_gtbp0, num_prebp0, num_gtbp0)

    for celue_id0 in gt_celue_dic:
        if celue_id0 not in preb_celue_dic:
            preb_celue_dic[celue_id0]=0
        if celue_id0 not in pre_celue_dic:
            pre_celue_dic[celue_id0]=0
        if celue_id0 in gt_celue_dic and celue_id0 in pre_celue_dic and celue_id0 in gtb_celue_dic and celue_id0 in preb_celue_dic:
            num_gtp0 = num_gtp0+gt_celue_dic[celue_id0]
            num_prep0 = num_prep0+pre_celue_dic[celue_id0]
            num_gtbp0 = num_gtbp0+gtb_celue_dic[celue_id0]
            num_prebp0 = num_prebp0+preb_celue_dic[celue_id0]
            re = pre_celue_dic[celue_id0]/gt_celue_dic[celue_id0]
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(celue_id0, gt_celue_dic[celue_id0]+gtb_celue_dic[celue_id0], gt_celue_dic[celue_id0]/(gt_celue_dic[celue_id0]+gtb_celue_dic[celue_id0]),
                                                      pre_celue_dic[celue_id0]/gt_celue_dic[celue_id0], preb_celue_dic[celue_id0]/gtb_celue_dic[celue_id0], pre_celue_dic[celue_id0], gt_celue_dic[celue_id0]))


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
    for idx, item in enumerate(sorted_data[:topn]):
        print(idx, item)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="./eval/vqav2")
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--mejson', type=str, default="")
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--precision', type=float, default=0.900)
    return parser.parse_args()

def main():
    args = parse_args()

    results, test_split, datalist = read_data(args)

    all_answers = []
    y_id = []
    y_scores = []
    y_true = []
    gt_label_dic = {0:0, 1:0, 2:0}
    pre_label_dic = gt_label_dic
 
    gt_celue_dic = {}
    gtb_celue_dic = {}

    multi_scores = []
    y_multi_true = []
    y_audit_label = []
    pre_celue_dic = {}
    preb_celue_dic = {}

    num_label = len(gt_label_dic)
    
    errinfo_dict = []
    hitinfo_dict=[]
    errbaiinfo_dict = []
   
    err99_file = open(args.mejson.replace(".jsonl", "") + "_err99.txt", 'w')
    errbai99_file = open(args.mejson.replace(".jsonl", "") + "_errbai99.txt", 'w')
    hit_file = open(args.mejson.replace(".jsonl", "") + "_hit.txt", 'w')

    for x in test_split:
        if x['id'] in results:
            all_answers.append({
                'question_id': x['id'],
                'answer': ''
            })

            score = results[x['id']]
            label = x['label']
            celue = x['celue']
            p = celue.find("/")
            if p != -1:
                celueid = celue[:p]
            else:
                celueid = celue

            if label=="100":
                lab=0
            else:
                lab=1
            mlab = get_label_type(label, gt_label_dic)
            y_true.append(lab)
            audit_label = label
            scoretensor = torch.tensor(score)
            y_scores.append(1-scoretensor[0])
            y_audit_label.append(label)

            multi_scores.append(score)
            y_multi_true.append(mlab)

            y_id.append(x['id'])

            if audit_label != '100':
                if celueid not in gt_celue_dic:
                    gt_celue_dic[celueid] = 1
                else:
                    gt_celue_dic[celueid] += 1
            else:
                if celueid not in gtb_celue_dic:
                    gtb_celue_dic[celueid] = 1
                else:
                    gtb_celue_dic[celueid] += 1

    for label in range(num_label):
        y_true_idx = [ int(j==label) for j in y_multi_true ]
        y_scores_idx = [j[label] for j in multi_scores]
        precision, recall, thresholds = precision_recall_curve(y_true_idx, y_scores_idx)
        get_best_f1(precision, recall, thresholds, topn=1)

        # 找到最接近0.99阈值的下标0
        precision_values = [0.990, 0.980, 0.950, 0.900]
        precision_indices = {value: 0 for value in precision_values}
        for i, v in enumerate(precision):
            for value in precision_values:
                if v <= value:
                    precision_indices[value] = i
        print("类别: ",  label, " precision阈值: ", " ".join(map(str, precision_values)))
        print("对应的位置:", " ".join("{:.6f}".format(thresholds[i]) for i in precision_indices.values()))
    # pdb.set_trace()

    confusion = torch.zeros(2, 2, dtype=torch.long)
    mconfusion = torch.zeros(num_label, num_label, dtype=torch.long)
    #tmp_idx = recall_indices[args.precision]
    tmp_idx = precision_indices[args.precision]
    # th=thresholds[tmp_idx]
    th = 0.0001
    for i, score in enumerate(y_scores):
        id = y_id[i]
        celue = datalist[id]['celue']
        p = celue.find("/")
        if p != -1:
            celueid = celue[:p]
        else:
            celueid = celue
        audit_label = y_audit_label[i]

        mscore=multi_scores[i]
        scoretensor = torch.tensor(mscore)
        max_probs, predict_label = torch.max(scoretensor, dim=-1)
        max_probs=float(max_probs)
        pre_mlabel = int(predict_label)
        # pre_mlabel = ans_dict[resultstext[id][0]]
        # th=0.5
        # th=0.96
        #if score >= th :
        if (pre_mlabel == 1 and max_probs > 0.950):
            lab=get_label_type(audit_label, pre_label_dic)
            pre_label=1
            if lab != pre_mlabel:
                if celueid not in pre_celue_dic:
                    pre_celue_dic[celueid] = 1
                else:
                    pre_celue_dic[celueid] += 1

                celue=datalist[id]['celue']
                # pdb.set_trace()
                qs = datalist[id]["input"].split("\n")
                pinlun=qs[1].split("评论:")[1]
                beijing=qs[0].split("视频背景信息:")[1]
                url = datalist[id]['image'].replace("/apdcephfs_qy3/share_301069248/data/video","http://9.22.25.210/xiaoshijie")
                res=str(id)+"\t"+str(max_probs)+"\t"+str(pre_mlabel)+"\t"+str(lab)+ "\t" + +"\t"+celue+"\t"+pinlun+"\t"+beijing+"\t"+url+"\n"
                errinfo_dict.append(res)
            else:
                if celueid not in preb_celue_dic:
                    preb_celue_dic[celueid] = 1
                else:
                    preb_celue_dic[celueid] += 1
        else:
            pre_label=0
            pre_mlabel=0
            if audit_label != '100':
                tmp_label = get_label_type(audit_label)
                celue=datalist[id]['celue']
                # pdb.set_trace()
                qs = datalist[id]["input"].split("\n")
                pinlun=qs[1].split("评论:")[1]
                beijing=qs[0].split("视频背景信息:")[1]
                url = datalist[id]['image'].replace("/apdcephfs_qy3/share_301069248/data/video","http://9.22.25.210/xiaoshijie")
                res=str(id)+"\t"+str(max_probs)+"\t"+str(pre_mlabel)+"\t"+str(lab)+"\t"+celue+"\t"+pinlun+"\t"+beijing+"\t"+url+"\n"
                errbaiinfo_dict.append(res)
        confusion[pre_label, y_true[i]] += 1
        mconfusion[pre_mlabel,y_multi_true[i]]+=1


    hit_file.writelines(hitinfo_dict)
    err99_file.writelines(errinfo_dict)
    errbai99_file.writelines(errbaiinfo_dict)

    print("pre_label_dic: ", pre_label_dic)
    print("gt_label_dic: ", gt_label_dic)
    print("\nMulti Confusion matrix:")
    print(mconfusion)
    eps = 1e-12
    jiangliang = confusion[0,:].sum().item() / (confusion.sum().item() + eps)
    daji = 1 - jiangliang
    # wusha =  confusion[1,0].item() / (confusion[1, :].sum().item() + eps)
    wusha = confusion[1,0].item() / (confusion[: , 0].sum().item() + eps)
    print("threshold {:.6f} jiangliang {:.4f} daji {:.4f} wusha {:.4f}".format(th, jiangliang, daji, wusha))
    print('准确率: ', confusion[1][1].item() / (confusion[1, :].sum().item() + eps) )
    print('黑样本覆盖率', confusion[1][1].item() /(confusion[: , 1].sum().item() + eps) )
    # print("Report precision, recall, and f1:")
    for i in range(mconfusion.size()[0]):
        p = mconfusion[i, i].item() / (mconfusion[i, :].sum().item() + eps)
        r = mconfusion[i, i].item() / (mconfusion[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        print("Label {:d},{:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
        # print("Label {}: {:.3f},{:.3f}, {:.3f}, {:.3f}".format(i, p, p2, r, f1))


    for i in range(11):
        if i in pre_label_dic:
            print(i, pre_label_dic[i], gt_label_dic[i])

    # statistic_celue_results(gt_celue_dic, pre_celue_dic, gtb_celue_dic, preb_celue_dic)


if __name__ == '__main__':
    main()
