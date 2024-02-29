import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(font='Franklin Gothic Book',
rc={
'axes.axisbelow': False,
'axes.edgecolor': 'lightgrey',
'axes.facecolor': 'None',
'axes.grid': False,
'axes.labelcolor': 'black',
'axes.spines.right': False,
'axes.spines.top': False,
'figure.facecolor': 'white',
'lines.solid_capstyle': 'round',
'patch.edgecolor': 'w',
'patch.force_edgecolor': True,
'text.color': 'dimgrey',
'xtick.bottom': False,
'xtick.color': 'dimgrey',
'xtick.direction': 'out',
'xtick.top': False,
'ytick.color': 'dimgrey',
'ytick.direction': 'out',
'ytick.left': False,
'ytick.right': False})
sns.set_context("notebook", rc={"font.size":16,
"axes.titlesize":20,
"axes.labelsize":18})


plt.xlim(0,2800) #x轴坐标轴
plt.ylim((0.84, 0.905))#y轴坐标轴
plt.xlabel('Fps')#x轴标签
plt.ylabel('S-measure')#y轴标签
BASNet = ['BASNet', 94, 0.8466, 87.06]
U2Net = ['U2Net', 128, 0.8541, 44.01]
SUCA = ['SUCA', 220 ,0.8968, 115.58]
EDNNet = ['EDN', 106 ,0.8944, 21.83]
EDRNet = ['EDRNet', 126 ,0.8651, 39.3]
DACNet = ['DACNet', 87 ,0.8855, 98.4]
EAMINet = ['EAMI', 90 ,0.8902, 99.1]
CSEPNet = ['CSEP', 98 ,0.8977, 18.78]
LFRNet = ['LFRNet', 206 ,0.8926, 18.85]
A3Net = ['A3Net', 160 ,0.9022, 16.98]
CSNet = ['CSNet', 744 ,0.877, 0.14]
HVPNet = ['HVPNet', 1017 ,0.8934, 1.23]
SAMNet = ['SAMNet', 970 ,0.8886, 1.33]
EDNLNet = ['EDN-Lite', 848 ,0.883, 1.8]
FSMINet = ['FSMI', 94 ,0.8869, 3.56]
Ours = ['Ours', 2756 ,0.9028, 0.93]
total = [BASNet, U2Net, SUCA, EDNNet, EDRNet, DACNet, EAMINet, CSEPNet, LFRNet, A3Net, CSNet, HVPNet, SAMNet, EDNLNet, FSMINet, Ours]
colors = np.random.rand(50)
for i, net in enumerate(total):
    #color = plt.cm.Set1(i)
    color = plt.cm.tab20b(colors[i])
    plt.scatter(net[1], net[2], net[3]*20, c=color,alpha=0.6)
    plt.text(x=net[1]+30,  # 文本x轴坐标
             y=net[2]-0.0006,  # 文本y轴坐标
             s=net[0],  # 文本内容
             fontdict=dict(fontsize=16,c = 'black', family='monospace',weight = 'bold'))

plt.scatter(1200, 0.845, 5*20, c='b',alpha=0.6)
plt.text(x=1200-250,  # 文本x轴坐标
         y=0.845-0.004,  # 文本y轴坐标
         s='Param: 5M',  # 文本内容
         fontdict=dict(fontsize=16,c = 'black', family='monospace',weight = 'bold'))

plt.scatter(1400, 0.845, 10*20, c='b',alpha=0.6)
plt.text(x=1400-40,  # 文本x轴坐标
         y=0.845-0.004,  # 文本y轴坐标
         s='10M',  # 文本内容
         fontdict=dict(fontsize=16,c = 'black', family='monospace',weight = 'bold'))
plt.scatter(1600, 0.845, 50*20, c='b',alpha=0.6)
plt.text(x=1600-40,  # 文本x轴坐标
         y=0.845-0.004,  # 文本y轴坐标
         s='50M',  # 文本内容
         fontdict=dict(fontsize=16,c = 'black', family='monospace',weight = 'bold'))
plt.show()
