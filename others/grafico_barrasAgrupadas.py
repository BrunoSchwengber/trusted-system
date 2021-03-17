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

#df = pd.read_csv(sys.argv[1], header=1, sep=';')
#print(df)

acuraciasB=[0.7955663467880772,0.8425053298634084,0.8559563392169229,0.859432157001267, 0.8672695429052107]
precisoesB=[0.9835915691381423,0.9674103871608861,0.9503879157137195,0.9399581294547347, 0.9318497941769939]
recallsB=[0.7955663467880772,0.8425053298634084,0.8559563392169229,0.859432157001267, 0.8672695429052107]

acuraciasP=[0.8672650700652887, 0.8672923879777616, 0.8701420702387956, 0.8696701557708321, 0.8710066822319562]
precisoesP=[0.9318497941769939, 0.9309001750618761, 0.93045207951399, 0.9278408408425793, 0.9273832751074604]
recallsP=[0.8672650700652887, 0.8672923879777616, 0.8701420702387956, 0.8696701557708321, 0.8710066822319562]

acuraciasT=[0.8672607996996019, 0.8672591430922233, 0.8672695797187081, 0.8672667082659187, 0.8672757275727573]
precisoesT=[0.9318308980277035, 0.9318308980277035, 0.9318497941769939, 0.9318497941769939, 0.9318500880752557]
recallsT=[0.8672607996996019, 0.8672591430922233, 0.8672695797187081, 0.8672667082659187, 0.8672757275727573]

acuraciasD=[0.878546785286778, 0.8613725510086767, 0.8526715054745143, 0.8285643855627016, 0.8070148342767487]
precisoesD=[0.9285394685330459, 0.930395100780088, 0.9376972362665021, 0.9337975521348115, 0.9327410614538653]
recallsD=[0.878546785286778, 0.8613725510086767, 0.8526715054745143, 0.8285643855627016, 0.8070148342767487]

acuraciasF=[0.8785456863262256, 0.8676152965235174, 0.99]
precisoesF=[0.9285394811003651, 0.929885556204196, 0.99]
recallsF=[0.8785456863262256, 0.8676152965235174, 0.99]
fscores=[0.8893, 0.8756, 0.99]

barwidth=0.20
print(fscores)
r1 = np.arange(len(acuraciasF))
r2 = [x + barwidth for 	x in r1]
r3 = [x + barwidth for 	x in r2]
r4 = [x + barwidth for 	x in r3]

f = plt.figure()


plt.bar(r1, acuraciasF, color='r', width=barwidth, label='Acurácia')
plt.bar(r2, precisoesF, color='blue', width=barwidth, label='Precisão')
plt.bar(r3, recallsF, color='g', width=barwidth, label='Recall')
plt.bar(r4, fscores, color='orange', width=barwidth, label='F1-Score')


plt.xlabel('Métodos')
plt.xticks([r + barwidth for r in range(len(acuraciasF))], ['ARF', 'FIXO', 'ARF'])
plt.ylabel('%')

plt.grid(True, axis='y', which='minor')
 
#plt.plot(x, df.iloc[:,2], 'orange', linestyle=':', label='ÁRVORE HOEFFDING', marker="*") 
#plt.plot(x, df.iloc[:,3], 'r', linestyle='--', label='ADWIN', marker=".")
#plt.plot(x, df.iloc[:,0], 'blue', linestyle='-.', label='D3', marker="s") 
#plt.plot(x, df.iloc[:,1], 'g', linestyle='-', label='D3 ADAPTADO', marker="v") 



#f.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
#          ncol=2, fancybox=True, shadow=True)
plt.legend(loc='lower center', ncol=2)#, fontsize=30)
plt.ylim(0,1)

f.savefig("compFinal.png", bbox_inches='tight')
plt.show()