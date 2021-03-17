import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.markers
import sys
import numpy as np
import statistics
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.figsize'] = (12,10)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 36

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{0:.0f}'.format(height*100),
                    #xy=(rect.get_x() + rect.get_width() / 2, height),
                    xy=(rect.get_x() + rect.get_width() / 2, 0.5),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=32)


#df = pd.read_csv(sys.argv[1], header=1, sep=';')
#print(df)
dataset = str(sys.argv[4])
entrada1=pd.read_csv(sys.argv[1])
entrada2=pd.read_csv(sys.argv[2])
entrada3=pd.read_csv(sys.argv[3])
#entrada4=pd.read_csv(sys.argv[4])
#entrada5=pd.read_csv(sys.argv[5])

acuracias = []
precisoes = []
recalls = []
fscores = []
am =[]
ad =[]
pm =[]
ppd =[]
rm =[]
rd =[]
fm =[]
fd = []

acuracias.append(entrada1.iloc[:,0])
acuracias.append(entrada2.iloc[:,0])
acuracias.append(entrada3.iloc[:,0])
#acuracias.append(entrada4.iloc[:,0])
#acuracias.append(entrada5.iloc[:,0])

precisoes.append(entrada1.iloc[:,1])
precisoes.append(entrada2.iloc[:,1])
precisoes.append(entrada3.iloc[:,1])
#precisoes.append(entrada4.iloc[:,1])
#precisoes.append(entrada5.iloc[:,1])

recalls.append(entrada1.iloc[:,2])
recalls.append(entrada2.iloc[:,2])
recalls.append(entrada3.iloc[:,2])
#recalls.append(entrada4.iloc[:,2])
#recalls.append(entrada5.iloc[:,2])

fscores.append(entrada1.iloc[:,3])
fscores.append(entrada2.iloc[:,3])
fscores.append(entrada3.iloc[:,3])
#fscores.append(entrada4.iloc[:,3])
#fscores.append(entrada5.iloc[:,3])

for i in range(0, 3, 1):
	am.append(statistics.mean(acuracias[i]))
	pm.append(statistics.mean(precisoes[i]))
	rm.append(statistics.mean(recalls[i]))
	fm.append(statistics.mean(fscores[i]))

for i in range(0, 3, 1):
	ad.append(statistics.stdev(acuracias[i]))
	ppd.append(statistics.stdev(precisoes[i]))
	rd.append(statistics.stdev(recalls[i]))
	fd.append(statistics.stdev(fscores[i]))

barwidth=0.20

r1 = np.arange(len(am))
r2 = [x + barwidth for 	x in r1]
r3 = [x + barwidth for 	x in r2]
r4 = [x + barwidth for 	x in r3]

f = plt.figure()

print(am, ad)

rec1 = plt.bar(r1, am, color='r', width=barwidth, label='Accuracy', yerr=ad)
rec2 = plt.bar(r2, pm, color='blue', width=barwidth, label='Precision', yerr=ppd)
rec3 = plt.bar(r3, rm, color='g', width=barwidth, label='Recall', yerr=rd)
rec4 = plt.bar(r4, fm, color='orange', width=barwidth, label='F1-Score', yerr=fd)


plt.xlabel('Systems')
plt.xticks([r + barwidth for r in range(3)], ['TRUSTED\nw/ HAC', 'FIXED\nw/ HAC', ' TRUSTED\nw/ ARF'])
plt.ylabel('%')

plt.grid(True, axis='y')

autolabel(rec1)
autolabel(rec2)
autolabel(rec3)
autolabel(rec4)

#f.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01),
#          ncol=2, fancybox=True, shadow=True)
plt.legend(loc='lower center', ncol=2)#, fontsize=30)
plt.ylim(0,1.1)

f.savefig("compFinal{}.pdf".format(dataset), bbox_inches='tight')
f.savefig("compFinal{}.png".format(dataset), bbox_inches='tight')
f.savefig("eps/compFinal{}.eps".format(dataset), bbox_inches='tight')
#plt.show()