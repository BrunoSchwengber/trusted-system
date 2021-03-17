import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.markers
import sys
import numpy as np
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.figsize'] = (12,10)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 36

def window_average(x,N):
    low_index = 0
    high_index = low_index + N
    w_avg = []
    while(high_index<len(x)):
        temp = sum(x[low_index:high_index])/N
        w_avg.append(temp)
        low_index = low_index + N
        high_index = high_index + N
    return w_avg

trusted = pd.read_csv(sys.argv[1])
fixo = pd.read_csv(sys.argv[2])
arf = pd.read_csv(sys.argv[3])
dataset = str(sys.argv[4])


x = np.linspace(0, 100, len(trusted.iloc[:,0]), endpoint=True)
y = np.linspace(0, 100, len(fixo.iloc[:,0]), endpoint=True)
z = np.linspace(0, 100, len(arf.iloc[:,0]), endpoint=True)


f = plt.figure()

 
#plt.plot(x, df.iloc[:,2], 'orange', linestyle=':', label='ÁRVORE HOEFFDING', marker="*") 
#plt.plot(z, arf.iloc[:,0], 'r', linestyle='--', label='TRUSTEC w/ ARF', marker=".")
#plt.plot(y, fixo.iloc[:,0], 'blue', linestyle='-.', label='FIXED w/ HAC', marker="s") 
#plt.plot(x, trusted.iloc[:,0], 'g', linestyle='-', label='TRUSTED w/ HAC', marker="v")
plt.plot(z, arf.iloc[:,0], 'r', linestyle='--', label='TRUSTED c/ ARF', marker=".")
plt.plot(y, fixo.iloc[:,0], 'blue', linestyle='-.', label='FIXO c/ HAC', marker="s") 
plt.plot(x, trusted.iloc[:,0], 'g', linestyle='-', label='TRUSTED c/ HAC', marker="v") 

plt.xlabel('\nPorcentagem dos Dados')
plt.ylabel('Acurácia')
#plt.xlabel('\nPercent of Data')
#plt.ylabel('Accuracy')

plt.grid(True, axis='y')

plt.legend(loc='lower left', fontsize=30)
plt.ylim(0,1.1)

f.savefig("acuracias{}_Port.pdf".format(dataset), bbox_inches='tight')
f.savefig("acuracias{}_Port.png".format(dataset), bbox_inches='tight')
f.savefig("eps/acuracias{}_Port.eps".format(dataset), bbox_inches='tight')
#plt.show()