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

df = pd.read_csv(sys.argv[1], header=0, sep=',')
#print(df)


x = np.linspace(0, 100, len(df.iloc[:,0]), endpoint=True)

f = plt.figure()

 
#plt.plot(x, df.iloc[:,2], 'orange', linestyle=':', label='ÁRVORE HOEFFDING', marker="*") 
plt.plot(x, df.iloc[:,0], 'r', linestyle='--', label='ARF', marker=".")
plt.plot(x, df.iloc[:,1], 'blue', linestyle='-.', label='FIXO', marker="s") 
plt.plot(x, df.iloc[:,2], 'g', linestyle='-', label='TRUSTED', marker="v") 

plt.xlabel('Porcentagem dos Dados')
plt.ylabel('Acurácia')

plt.grid(True, axis='y')

plt.legend(loc='lower left', fontsize=30)
plt.ylim(0,1.01)

f.savefig("acuracias-comp.pdf", bbox_inches='tight')
f.savefig("acuracias-comp.png", bbox_inches='tight')
plt.show()