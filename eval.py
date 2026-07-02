#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
import argparse
from collections import defaultdict

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends

def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float16)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        if max(m_row, n_col) == 0:
            score = 0
        else:
            score = (1 - D[-1, -1] / max(m_row, n_col)) * 100

    else:
        score = D[-1, -1]

    return score

def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)

def f_score(recognized, ground_truth, overlap, bg_class=["background"], category_hits=None):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0
    fn = 0

    hits = np.zeros(len(y_label))

    # Category-wise precision, recall, F1 score
    category_tp = defaultdict(int)
    category_fp = defaultdict(int)
    category_fn = defaultdict(int)

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])

        if IoU.size == 0:
            continue

        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
            category_tp[y_label[idx]] += 1
        else:
            fp += 1
            category_fp[p_label[j]] += 1

    fn = len(y_label) - sum(hits)
    for i in range(len(y_label)):
        if not hits[i]:
            category_fn[y_label[i]] += 1

    if category_hits is not None:
        for category, value in category_tp.items():
            category_hits['tp'][category] += value
        for category, value in category_fn.items():
            category_hits['fn'][category] += value

    # Calculate Precision, Recall, F1 per category
    category_metrics = {}
    for category in category_tp:
        precision = category_tp[category] / (category_tp[category] + category_fp[category]) if (category_tp[category] + category_fp[category]) > 0 else 0
        recall = category_tp[category] / (category_tp[category] + category_fn[category]) if (category_tp[category] + category_fn[category]) > 0 else 0
        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        category_metrics[category] = {'precision': precision, 'recall': recall, 'f1': f1}

    return float(tp), float(fp), float(fn), category_metrics

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="monkey")
    parser.add_argument('--split', default='1')
    parser.add_argument('--recog_path', default=None)

    args = parser.parse_args()

    ground_truth_path = "./data/" + args.dataset + "/groundTruth/"
    # ground_truth_path = "./data/" + args.dataset + "/gt/"
    recog_path = args.recog_path.rstrip("/\\") + "/" if args.recog_path else "./results/" +  args.dataset + "/split_" + args.split + "/"
    file_list = "./data/" + args.dataset + "/splits/test_split_" + args.split + ".bundle"

    list_of_videos = read_file(file_list).split('\n')[:-1]

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0

    category_hits = {'tp': defaultdict(int), 'fn': defaultdict(int)}

    all_category_metrics = defaultdict(list)  # Store category metrics for all videos

    for vid in list_of_videos:
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]

        recog_file = recog_path + vid.split('.')[0]+".txt"
        recog_content = read_file(recog_file).split('\n')[1].split()

        if len(gt_content) != len(recog_content):
            raise ValueError("Prediction length mismatch for %s: gt=%d pred=%d" % (vid, len(gt_content), len(recog_content)))

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1, category_metrics = f_score(recog_content, gt_content, overlap[s], category_hits=category_hits)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
            # Store category metrics (precision, recall, F1)
            for category, metrics in category_metrics.items():
                all_category_metrics[category].append(metrics)

    print("Acc: %.4f" % (100 * float(correct) / total))
    print('Edit: %.4f' % ((1.0 * edit) / len(list_of_videos)))

    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])

        f1 = 2.0 * (precision * recall) / (precision + recall)

        f1 = np.nan_to_num(f1) * 100
        print('F1@%0.2f: %.4f' % (overlap[s], f1))

    print("\nCategory-wise Recall:")
    for category in category_hits['tp']:
        recall = category_hits['tp'][category] / float(category_hits['tp'][category] + category_hits['fn'][category])
        print("Category: %s, Recall: %.4f" % (category, recall))

    # Print category-wise metrics (Precision, Recall, F1) for all videos
    print("\nCategory-wise Precision, Recall, F1 scores:")
    for category, metrics_list in all_category_metrics.items():
        avg_precision = np.mean([metrics['precision'] for metrics in metrics_list])
        avg_recall = np.mean([metrics['recall'] for metrics in metrics_list])
        avg_f1 = np.mean([metrics['f1'] for metrics in metrics_list])
        print("Category: %s, Precision: %.4f, Recall: %.4f, Avg F1: %.4f" % (category, avg_precision, avg_recall, avg_f1))

if __name__ == '__main__':
    main()
