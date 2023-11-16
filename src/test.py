import numpy as np
import dill
import torch
import time
import argparse
from models import AllRecDrugModel
from util import llprint, multi_label_metric, ddi_rate_score
import torch.nn.functional as F



data_path = "../data/output/records_final.pkl"
voc_path = "../data/output/voc_final.pkl"
ddi_adj_path = "../data/output/ddi_A_final.pkl"
ehr_adj_path = '../data/output/ehr_adj_final.pkl'
ddi_mask_path = '../data/output/ddi_mask_H.pkl'
device = torch.device("cuda:0")
ehr_adj = dill.load(open(ehr_adj_path, 'rb'))   
ddi_adj = dill.load(open(ddi_adj_path, "rb"))
ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))
data = dill.load(open(data_path, "rb"))
voc = dill.load(open(voc_path, "rb"))
diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

split_point = int(len(data) * 2 / 3)
data_train = data[:split_point]
eval_len = int(len(data[split_point:]) / 2)
data_eval = data[split_point : split_point + eval_len]
data_test = data[split_point + eval_len :] 
voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

def eval(model, data_eval, voc_size, epoch):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input):
            target_output, _  = model(input[: adm_idx + 1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path="../data/output/ddi_A_final.pkl")

    llprint(
        "\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )

model = AllRecDrugModel(
        voc_size,
        ehr_adj,
        ddi_adj,
        ddi_mask_H,
        fai = 0.01,
        emb_dim=64,
        device=device,
    )

resume_path = 'saved/best.model'
parser = argparse.ArgumentParser()
parser.add_argument("--resume_path", type=str, default=resume_path, help="resume path")
parser.add_argument("--seed", type=int, default=8, help="resume path")
args = parser.parse_args()
np.random.seed(args.seed)
model.load_state_dict(torch.load(open(args.resume_path, "rb"), map_location = torch.device('cpu')),False)

model.to(device=device)
tic = time.time()
result = []
for _ in range(10):
    test_sample = np.random.choice(
        data_test, round(len(data_test) * 0.8), replace=True
    )
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
        model, test_sample, voc_size, 0
    )
    result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
result = np.array(result)
mean = result.mean(axis=0)
std = result.std(axis=0)
outstring = ""
for m, s in zip(mean, std):
    outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)
print(outstring)
print("test time: {}".format(time.time() - tic))