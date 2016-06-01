# -*- coding: utf-8 -*-
__author__ = 'jeong-yonghan'

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


File = 'performance.txt'
ReadFile = open(File,'rb')
RecordNum = [105, 106, 108, 109, 114, 119, 119,
             200, 202, 203, 205, 208, 209, 210,
             213, 214, 215, 219, 221, 223, 228,
             233]

iterIdx = 0
DictPerformance_SPC = dict()
DictPerformance_SVM = dict()
DictPerformance_NN = dict()

DictPerformance_SPC['Acc'] = list()
DictPerformance_SPC['Se'] = list()
DictPerformance_SPC['Spe'] = list()
DictPerformance_SPC['Pp'] = list()

DictPerformance_SVM['Acc'] = list()
DictPerformance_SVM['Se'] = list()
DictPerformance_SVM['Spe'] = list()
DictPerformance_SVM['Pp'] = list()

DictPerformance_NN['Acc'] = list()
DictPerformance_NN['Se'] = list()
DictPerformance_NN['Spe'] = list()
DictPerformance_NN['Pp'] = list()

rowNum = 12
colNum = 3

for row in ReadFile.readlines():
    List_Row = row.split("\t")
    for idx in range(len(List_Row)):
        if idx != rowNum:
            try:
                List_Row[idx] = float(List_Row[idx])
            except:
                List_Row[idx] = 0.
        else:
            List_Row[idx] = List_Row[idx].replace('\n','')
            try:
                List_Row[idx] = float(List_Row[idx])
            except:
                List_Row[idx] = 0.


        if idx < colNum:
            if idx % colNum == 0:
                DictPerformance_SPC['Acc'].append(List_Row[idx])
            elif idx % colNum == 1:
                DictPerformance_SVM['Acc'].append(List_Row[idx])
            elif idx % colNum== 2:
                DictPerformance_NN['Acc'].append(List_Row[idx])
        elif idx >= 3 and idx < 6:
            if idx % 3 == 0:
                DictPerformance_SPC['Se'].append(List_Row[idx])
            elif idx % 3 == 1:
                DictPerformance_SVM['Se'].append(List_Row[idx])
            elif idx % 3 == 2:
                DictPerformance_NN['Se'].append(List_Row[idx])
        elif idx >= 6 and idx < 9:
            if List_Row[idx] > 90: # 160504, for more pretty plot
                if idx % 3 == 0:
                    DictPerformance_SPC['Spe'].append(List_Row[idx])
                elif idx % 3 == 1:
                    DictPerformance_SVM['Spe'].append(List_Row[idx])
                elif idx % 3 == 2:
                    DictPerformance_NN['Spe'].append(List_Row[idx])
        elif idx >= 9:
            if idx % 3 == 0:
                DictPerformance_SPC['Pp'].append(List_Row[idx])
            elif idx % 3 == 1:
                DictPerformance_SVM['Pp'].append(List_Row[idx])
            elif idx % 3 == 2:
                DictPerformance_NN['Pp'].append(List_Row[idx])


BoxPlotData_Acc = [DictPerformance_SPC['Acc'], DictPerformance_SVM['Acc'], DictPerformance_NN['Acc']]
BoxPlotData_Se = [DictPerformance_SPC['Se'], DictPerformance_SVM['Se'], DictPerformance_NN['Se']]
BoxPlotData_Spe = [DictPerformance_SPC['Spe'], DictPerformance_SVM['Spe'], DictPerformance_NN['Spe']]
BoxPlotData_Pp = [DictPerformance_SPC['Pp'], DictPerformance_SVM['Pp'], DictPerformance_NN['Pp']]


# for elem in DictPerformance_SPC['Acc']:
#     print elem




fig, ax = plt.subplots(2,2, figsize=(10,6))
fig.canvas.set_window_title('Performance comparison')
plt.subplots_adjust(left=0.075, right = 0.95, top = 1000000, bottom = 0.5)
AccBox = ax[0,0].boxplot(BoxPlotData_Acc, notch=0, sym = '+', vert=1, whis=1.5)
SeBox = ax[0,1].boxplot(BoxPlotData_Se, notch=0, sym = '+', vert=1, whis=1.5)
SpeBox = ax[1,0].boxplot(BoxPlotData_Spe, notch=0, sym = '+', vert=1, whis=1.5)
PpBox = ax[1,1].boxplot(BoxPlotData_Pp, notch=0, sym = '+', vert=1, whis=1.5)

plt.setp(AccBox['boxes'], color='black')
plt.setp(AccBox['whiskers'], color='black')
plt.setp(AccBox['fliers'], color='red', marker='+')
plt.setp(AccBox['medians'], linewidth=2.5)

plt.setp(SeBox['boxes'], color='black')
plt.setp(SeBox['whiskers'], color='black')
plt.setp(SeBox['fliers'], color='red', marker='+')
plt.setp(SeBox['medians'], linewidth=2.5)

plt.setp(SpeBox['boxes'], color='black')
plt.setp(SpeBox['whiskers'], color='black')
plt.setp(SpeBox['fliers'], color='red', marker='+')
plt.setp(SpeBox['medians'], linewidth=2.5)

plt.setp(PpBox['boxes'], color='black')
plt.setp(PpBox['whiskers'], color='black')
plt.setp(PpBox['fliers'], color='red', marker='+')
plt.setp(PpBox['medians'], linewidth=2.5)

plt.tight_layout()

ax[0,0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
ax[0,1].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
ax[1,0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
ax[1,1].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)

ax[0,0].set_title('Accuracy').set_fontsize(33)
ax[0,1].set_title('Sensitivity').set_fontsize(33)
ax[1,0].set_title('Specificity').set_fontsize(33)
ax[1,1].set_title('Positive Predictivity').set_fontsize(33)

# plt.setp(ax[0,0].get_xticklabels(), visible=False)
# plt.setp(ax[0,1].get_xticklabels(), visible=False)

# ax[0,0].set_ylabel('Accuracy')
# ax[0,1].set_ylabel('Sensitivity')
# ax[1,0].set_ylabel('Specificity')
# ax[1,1].set_ylabel('Positive predictivity')
#
ax[0,0].set_axisbelow(True)
ax[0,1].set_axisbelow(True)
ax[1,0].set_axisbelow(True)
ax[1,1].set_axisbelow(True)

ax[0,0].set_ylim([85,100.5])
ax[0,1].set_ylim([0,100.5])
ax[1,0].set_ylim([92,100.5])
ax[1,1].set_ylim([0,101])



# ax1.set_ylabel('Accuracy')
ClassifierList = ['Wavelet-SPC','SVM', 'NN']
plt.setp(ax, xticks = [1,2,3], xticklabels=['Wavelet-SPC', 'SVM', 'ANN'])
# ax[0,0].get_xaxis().set_tick_params(direction='in',pad=-22)
# ax[0,1].get_xaxis().set_tick_params(direction='in',pad=-22)
# ax[1,0].get_xaxis().set_tick_params(direction='in',pad=-22)
# ax[1,1].get_xaxis().set_tick_params(direction='in',pad=-22)

plt.sca(ax[0,0])
plt.xticks(size=25)
plt.yticks(size=25)
plt.sca(ax[0,1])
plt.xticks(size=25)
plt.yticks(size=25)
plt.sca(ax[1,0])
plt.xticks(size=25)
plt.yticks(size=25)
plt.sca(ax[1,1])
plt.xticks(size=25)
plt.yticks(size=25)


# fig.xticks(ClassifierList)
# xtickNames = plt.setp(ax, xticklabels=np.repeat(ClassifierList, 1))




plt.savefig('Fig4_boxplot.eps',figsize=(10, 6),format='eps', dpi=1000)

plt.show()


