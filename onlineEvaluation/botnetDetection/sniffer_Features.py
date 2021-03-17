from __future__ import print_function
from scapy.all import *
#import slowloris.py
import numpy as np
import collections
import csv
import netifaces
import threading
import time
import GenerateFlows as GF
from trusted import Trusted
from fixedAglo import Fixed
from arf import Arf
import datetime

global_lock = threading.Lock()
#############################################################################
#	SNIFFER DE REDE PARA EXTRACAO DE CARACTERISTICAS ONLINE         	    #
#			COM BASE EM ARQUIVO PCAP                                        #
#																			#		
#	AUTOR: BRUNO HENRIQUE SCHWENGBER										#						
#	DATA:  09/2020											     			#
#############################################################################

class Sniffer():
	cont=0
	entrada = []
	pkts = []
	dest = []
	sport = []
	interface = None
	entropias = []
	cont_detection = 0
	timeFirst = 0
	trusted = Trusted(0.1, 1000, 0.2, 0.7)
	fixed = Fixed(0.1)
	arf = Arf()
	thr_count=0
	threads = []

	def thread_train_test(self, flowFeatures):
		self.cont=0
		self.pkts.clear()
		resultados = []

		#acuracia_t, precisao_t, recall_t, fscore_t = self.trusted.process(flowFeatures)
		#resultados.append(['trusted', acuracia_t, precisao_t, recall_t, fscore_t])
		acuracia_f, precisao_f, recall_f, fscore_f = self.fixed.process(flowFeatures)
		resultados.append(['fixed', acuracia_f, precisao_f, recall_f, fscore_f])
		#acuracia_a, precisao_a, recall_a, fscore_a = self.arf.process(flowFeatures)
		#resultados.append(['arf', acuracia_a, precisao_a, recall_a, fscore_a])

		while global_lock.locked():
			time.sleep(0.1)
			continue

		global_lock.acquire()
		
		for resultado in resultados:
			arquivo = open("{}-base-45.csv".format(resultado[0]), 'a')
			print("------------------{}-------------------".format(resultado[0]))
			for i, j, k, l in zip(resultado[1], resultado[2], resultado[3], resultado[4]):
				string = str(i)+','+str(j)+','+str(k)+','+str(l)+'\n'
				print(resultado[0])
				print(string)
				arquivo.write(string)
			arquivo.close()
		print("----------------------------------------")
		global_lock.release()


	#Funcao responsavel por capturar o tamanho dos pacotes e adicionar
	#ao vetor entrada que deve ser passado para a funcao gerarFluxos()
	def pkt_handler(self,pkt): 
		#leng = pkt.len #captura o valor do tamanho do pacote e adiciona na variavel len
		self.cont += 1 #contadore e acrescentado em 1 para contagem do tamanho da janela
		self.pkts.append(pkt)
		#print(pkt.summary())

		if self.cont == 1:
			self.timeFirst=time.time()
			#print(datetime.datetime.now())

		if (time.time()-self.timeFirst>=20):
			self.thr_count += 1
			flowFeatures = GF.generateFlow(self.pkts)
			print("{} packets".format(self.cont))			
			print("{} rodadas de treino e teste".format(self.thr_count))
			Thread(target = self.thread_train_test, args = [flowFeatures]).start()
			#self.threads.append(t)

#Inicia o Codigo chamando a funcao principal.
if __name__ == '__main__': 
	snif = Sniffer()
	os.system('mkdir resultados_antigos')
	os.system('mv *.csv resultados_antigos/')
	sniff(iface='eth1', prn=snif.pkt_handler, filter="tcp", store=0)
			
