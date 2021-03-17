from __future__ import print_function
from scapy.all import *
#import slowloris.py
import numpy as np
import collections
import csv
import netifaces
import threading
import time
from collections import OrderedDict

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
	FIN = 0x01
	SYN = 0x02
	RST = 0x04
	PSH = 0x08
	ACK = 0x10
	URG = 0x20
	ECE = 0x40
	CWR = 0x80
	
	pcap = rdpcap(sys.argv[1])


	#Funcao responsavel por capturar o tamanho dos pacotes e adicionar
	#ao vetor entrada que deve ser passado para a funcao gerarFluxos()
	def find_response(self, p):
		print("SYN")
		#for pkt in pcap:
		print(p[TCP].seq)
		print(p[TCP].syn)

	def pkt_handler(self,p):
		print((TCP in p))
		if (TCP in p):
			print(p[IP].src)
			print(p[TCP].flags)
			print(p[TCP].seq)
			print(p[TCP].ack)

			self.flag_s.append(p[TCP].flags)

			self.flag_s = list(OrderedDict.fromkeys(self.flag_s))
			print(self.flag_s)
		elif (UDP in p):
			print("----------UDP------------------")
			print(p[IP].src)
			print("----------------------------------")
		else:
			print("-------------OUTRO-----------------")
			print(p.summary())			
			print("----------------------------------")

		#print('TCP')
		#if ('A' in p[TCP].flags):
		#	self.find_response(p)
		#	else:
		#		exit(0)
			

	
#Inicia o Codigo chamando a funcao principal.
if __name__ == '__main__': 
	snif = Sniffer()
	snif.flag_s=[]
	
	sniff(iface='eth1', prn=snif.pkt_handler, filter="", store=0)

			
