import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,precision_recall_curve,average_precision_score
import pandas as pd
import sys
import os
import os.path as osp
# get the path
script_dir, script_name = osp.split(os.path.abspath(sys.argv[0]))
parent_dir = osp.dirname(script_dir)
RESULT_BASE_PATH = parent_dir + '/result/'
roc_save_path = RESULT_BASE_PATH + 'ROC_PR/'
#绘制RPI2241特征组合ROC曲线
none_path = RESULT_BASE_PATH + 'ROC_PR' + '/RPI2241_none.csv'
none = pd.read_csv(none_path)
fpr_none, tpr_none, thresholds_none = roc_curve(none['label'], none['pred'])
auc_none = auc(fpr_none, tpr_none)
plt.plot(fpr_none, tpr_none, color='limegreen',lw =2.5,label='none (AUC = %0.4f)' % auc_none)
# plt.plot(fpr, tpr, color='darkorange', label='none (AUC = 0.9023)')
struc_path = RESULT_BASE_PATH + 'ROC_PR' + '/RPI2241_struc.csv'
struc = pd.read_csv(struc_path)
fpr_struc, tpr_struc, thresholds_struc = roc_curve(struc['label'], struc['pred'])
auc_struc = auc(fpr_struc, tpr_struc)
plt.plot(fpr_struc, tpr_struc, color='red', lw =2.5,label='struc (AUC = %0.4f)' % auc_struc)
seq_path = RESULT_BASE_PATH + 'ROC_PR' + '/RPI2241_seq.csv'
seq = pd.read_csv(seq_path)
fpr_seq, tpr_seq, thresholds_seq = roc_curve(seq['label'], seq['pred'])
auc_seq = auc(fpr_seq, tpr_seq)
plt.plot(fpr_seq, tpr_seq, color='deepskyblue',lw =2.5, label='seq (AUC = %0.4f)' % auc_seq)
seqstruc_path = RESULT_BASE_PATH + 'ROC_PR' + '/RPI2241_seqstruc.csv'
seqstruc = pd.read_csv(seqstruc_path)
fpr_seqstruc, tpr_seqstruc, thresholds_seqstruc = roc_curve(seqstruc['label'], seqstruc['pred'])
auc_seqstruc = auc(fpr_seqstruc, tpr_seqstruc)
plt.plot(fpr_seqstruc, tpr_seqstruc, color='violet',lw =2.5, label='seq+struc (AUC = %0.4f)' % auc_seqstruc)
# plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RPI2241 ROC and AUC')
plt.legend(loc="lower right")
plt.savefig(roc_save_path + 'RPI2241_fearoc.tif', dpi=500)
plt.show()
# #绘制RPI2241特征组合PR曲线
# none_path = RESULT_BASE_PATH + 'ROC_PR' + '/RPI2241_none.csv'
# none = pd.read_csv(none_path)
# precision_none, recall_none,thresholds_none = precision_recall_curve(none['label'], none['pred'])
# ap_none = average_precision_score(none['label'], none['pred'])
# plt.plot(recall_none,precision_none, color='limegreen',lw =2.5, label='none (AUPR = %0.4f)' % ap_none )
# struc_path = RESULT_BASE_PATH + 'ROC_PR' + '/RPI2241_struc.csv'
# struc = pd.read_csv(struc_path)
# precision_struc, recall_struc,thresholds_struc = precision_recall_curve(struc['label'], struc['pred'])
# ap_struc = average_precision_score(struc['label'], struc['pred'])
# plt.plot(recall_struc,precision_struc, color='red', lw =2.5,label='struc (AUPR = %0.4f)' % ap_struc )
# seq_path = RESULT_BASE_PATH + 'ROC_PR' + '/RPI2241_seq.csv'
# seq = pd.read_csv(seq_path)
# precision_seq, recall_seq,thresholds_seq = precision_recall_curve(seq['label'], seq['pred'])
# ap_seq = average_precision_score(seq['label'], seq['pred'])
# plt.plot(recall_seq,precision_seq, color='deepskyblue', lw =2.5,label='seq (AUPR = %0.4f)' % ap_seq )
# seqstruc_path = RESULT_BASE_PATH + 'ROC_PR' + '/RPI2241_seqstruc.csv'
# seqstruc = pd.read_csv(seqstruc_path)
# precision_seqstruc, recall_seqstruc,thresholds_seqstruc = precision_recall_curve(seqstruc['label'], seqstruc['pred'])
# ap_seqstruc = average_precision_score(seqstruc['label'], seqstruc['pred'])
# plt.plot(recall_seqstruc,precision_seqstruc, color='violet',lw =2.5, label='seq+struc (AUPR = %0.4f)' % ap_seqstruc )
# # plt.plot([0, 1], [1, 0], color='black', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.5, 1.05])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('RPI2241 Precision-Recall Curve')
# plt.legend(loc="lower left")
# #plt.savefig(roc_save_path + 'RPI2241_feapr.jpg', dpi=800)
# plt.show()
#绘制NPInter特征组合ROC曲线
none_path = RESULT_BASE_PATH + 'ROC_PR' + '/NPInter_none.csv'
none = pd.read_csv(none_path)
fpr_none, tpr_none, thresholds_none = roc_curve(none['label'], none['pred'])
auc_none = auc(fpr_none, tpr_none)
plt.plot(fpr_none, tpr_none, color='limegreen', lw=2,label='none (area = %0.4f)' % auc_none)
struc_path = RESULT_BASE_PATH + 'ROC_PR' + '/NPInter_struc.csv'
struc = pd.read_csv(struc_path)
fpr_struc, tpr_struc, thresholds_struc = roc_curve(struc['label'], struc['pred'])
auc_struc = auc(fpr_struc, tpr_struc)
plt.plot(fpr_struc, tpr_struc, color='red', lw=2,label='struc (area = %0.4f)' % auc_struc)
seq_path = RESULT_BASE_PATH + 'ROC_PR' + '/NPInter_seq.csv'
seq = pd.read_csv(seq_path)
fpr_seq, tpr_seq, thresholds_seq = roc_curve(seq['label'], seq['pred'])
auc_seq = auc(fpr_seq, tpr_seq)
plt.plot(fpr_seq, tpr_seq, color='deepskyblue', lw=2,label='seq (area = %0.4f)' % auc_seq)
seqstruc_path = RESULT_BASE_PATH + 'ROC_PR' + '/NPInter_seqstruc.csv'
seqstruc = pd.read_csv(seqstruc_path)
fpr_seqstruc, tpr_seqstruc, thresholds_seqstruc = roc_curve(seqstruc['label'], seqstruc['pred'])
auc_seqstruc = auc(fpr_seqstruc, tpr_seqstruc)
plt.plot(fpr_seqstruc, tpr_seqstruc, color='violet', lw=2,label='seq+struc (area = %0.4f)' % auc_seqstruc)
# plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('NPInter v2.0 ROC and AUC')
plt.legend(loc="lower right")
plt.savefig(roc_save_path + 'NPInter_fearoc.tif', dpi=500)
plt.show()

##放大局部，方便观察
#
# inset_ax.set_xlim([0.1,0.4])
# inset_ax.set_ylim([0.6,1.05])
# inset_ax.grid()
# plt.savefig(roc_save_path + 'NPInter_fearoc.jpg', dpi=800)
# plt.show()
# #绘制NPInter特征组合PR曲线
# none_path = RESULT_BASE_PATH + 'ROC_PR' + '/NPInter_none.csv'
# none = pd.read_csv(none_path)
# precision_none, recall_none,thresholds_none = precision_recall_curve(none['label'], none['pred'])
# ap_none = average_precision_score(none['label'], none['pred'])
# plt.plot(recall_none,precision_none, color='limegreen',lw=2, label='none (AUPR = %0.4f)' % ap_none )
# struc_path = RESULT_BASE_PATH + 'ROC_PR' + '/NPInter_struc.csv'
# struc = pd.read_csv(struc_path)
# precision_struc, recall_struc,thresholds_struc = precision_recall_curve(struc['label'], struc['pred'])
# ap_struc = average_precision_score(struc['label'], struc['pred'])
# plt.plot(recall_struc,precision_struc, color='red', lw=2,label='struc (AUPR = %0.4f)' % ap_struc )
# seq_path = RESULT_BASE_PATH + 'ROC_PR' + '/NPInter_seq.csv'
# seq = pd.read_csv(seq_path)
# precision_seq, recall_seq,thresholds_seq = precision_recall_curve(seq['label'], seq['pred'])
# ap_seq = average_precision_score(seq['label'], seq['pred'])
# plt.plot(recall_seq,precision_seq, color='deepskyblue', lw=2,label='seq (AUPR = %0.4f)' % ap_seq )
# seqstruc_path = RESULT_BASE_PATH + 'ROC_PR' + '/NPInter_seqstruc.csv'
# seqstruc = pd.read_csv(seqstruc_path)
# precision_seqstruc, recall_seqstruc,thresholds_seqstruc = precision_recall_curve(seqstruc['label'], seqstruc['pred'])
# ap_seqstruc = average_precision_score(seqstruc['label'], seqstruc['pred'])
# plt.plot(recall_seqstruc,precision_seqstruc, color='violet', lw=2,label='seq+struc (AUPR = %0.4f)' % ap_seqstruc )
# # plt.plot([0, 1], [1, 0], color='black', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.5, 1.05])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('NPInter v2.0 Precision-Recall Curve')
# plt.legend(loc="lower left")
# #plt.savefig(roc_save_path + 'NPInter_feapr.jpg', dpi=800)
# plt.show()

