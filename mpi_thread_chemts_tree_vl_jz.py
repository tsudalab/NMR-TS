from math import *
import random
import numpy as np
import random as pr
import time
import math
import shutil
import csv
import ast
import tensorflow as tf
import subprocess
from load_model import loaded_model
from keras.preprocessing import sequence
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles, MolFromSmarts
from rdkit.Chem import Crippen
import sys
from threading import Thread, Lock, RLock
import threading
from Queue import Queue
from mpi4py import MPI
from RDKitText import transfersdf
from SDF2GauInput import GauTDDFT_ForDFT
from GaussianRunPack import GaussianDFTRun
from guppy import hpy

import sascorer
from scipy.stats import wasserstein_distance
import ConfigParser

smiles_max_len = 81 # zinc dataset
state_length = 64


class chemical:

    def __init__(self):

        self.position=['&']
    def Clone(self):

        st = chemical()
        st.position= self.position[:]
        return st

    def SelectPosition(self,m):

        self.position.append(m)

    def Getatom(self):
        return [i for i in range(self.num_atom)]

class Node:

    def __init__(self, position = None, parent = None, state = None, nodelock=threading.Lock()):
        self.position = position
        self.parentNode = parent
        self.childNodes = []
        #self.child=None
        self.wins = 0
        self.re_max = 0
        self.visits = 0
        #self.depth=0
        #self.first_time_visit=1
        self.expanded=[]
        self.nodeadded=[]
        self.random_node=[]
        #self.all_posible=[]
        #self.generate_smile=[]
        #self.node_index=[]
        #self.valid_smile=[]
        #self.new_compound=[]
        #self.nodelock=nodelock
        #self.ucb=[]
        #self.core_id=[]
        self.virtual_loss=0
        self.num_thread_visited=0
        self.all_probs=[]


    def Selectnode(self, ts_strategy, search_parameter, alpha):
        #self.nodelock.acquire()

        ucb=[]
        ntv_list=[]
        base_list=[]
        bias_list=[]
        max_list=[]
        #print "current node's virtual_loss:",self.num_thread_visited,self.virtual_loss
        for i in range(len(self.childNodes)):
            #print "current node's childrens' virtual_loss:",self.childNodes[i].num_thread_visited,self.childNodes[i].virtual_loss
            C = search_parameter
            cNodei = self.childNodes[i]
            if ts_strategy == 'uct':
                ucb.append(alpha*(cNodei.wins)/(0.0001+cNodei.visits+cNodei.num_thread_visited)+
                           (1-alpha)*cNodei.re_max/(1+cNodei.num_thread_visited)+
                           C*sqrt(2*log(self.visits+self.num_thread_visited)/(0.0001+cNodei.visits+cNodei.num_thread_visited)))
            elif ts_strategy == 'puct':
                prob=self.all_probs[i]
                ucb.append(alpha*(cNodei.wins)/(0.001+cNodei.visits+cNodei.num_thread_visited)+
                           (1-alpha)*cNodei.re_max/(1+cNodei.num_thread_visited)+
                           C*(np.tanh(2*prob-1)+1)/2*sqrt((self.visits+self.num_thread_visited))/(1+cNodei.visits+cNodei.num_thread_visited))
            ntv_list.append(cNodei.num_thread_visited)
            base_list.append(alpha*(cNodei.wins)/(0.001+cNodei.visits+cNodei.num_thread_visited)+(1-alpha)*cNodei.re_max/(1+cNodei.num_thread_visited))
            bias_list.append(ucb[-1] - base_list[-1])
            max_list.append(cNodei.re_max)
        #print 'ucb score list', ucb
        #print 'ntv_list', ntv_list, 'cNodei.num_thread_visited', cNodei.num_thread_visited, 'total', np.sum(ntv_list)
        #print 'base_list', base_list
        #print 'bias_list', bias_list
        #print 'max_list', max_list

        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        ind=pr.choice(indices)
        s=self.childNodes[ind]
        #print "which thread's ucb:",threading.currentThread().getName()
        #print ucb
        #self.nodelock.release()
        return s

    def Addnode(self, m):

        #n = Node(position = m, parent = self, state = s)
        self.nodeadded.remove(m)
        n = Node(position = m, parent = self)
        self.childNodes.append(n)
        #print('This:',self.position,'Parent',self.parentNode,'position',self.position,'children',self.childNodes,'expanded',self.expanded,'added',self.nodeadded,'probs',self.all_probs)
        #if self.parentNode != None:
            #print('Parent:',self.parentNode.position)
        return n



    def Update(self, result, add_vis_count = 1):
        #self.nodelock.acquire()
        #print "update visits:",self.visits
        self.visits += add_vis_count
        self.wins += result
        if self.re_max < result:
            self.re_max = result
        #self.nodelock.release()

    def delete_virtual_loss(self):
        #self.num_thread_visited=0
        self.num_thread_visited += -1
        self.virtual_loss=0

    def expanded_node1(self, model, state, val):


        all_nodes=[]

        end="\n"
        position=[]
        position.extend(state)
        total_generated=[]
        new_compound=[]
        get_int_old=[]
        for j in range(len(position)):
            get_int_old.append(val.index(position[j]))

        get_int=get_int_old
        x=np.reshape(get_int,(1,len(get_int)))
        #x_pad= sequence.pad_sequences(x, maxlen=42, dtype='int32', padding='post', truncating='pre', value=0.) #original
        x_pad= sequence.pad_sequences(x, maxlen=82, dtype='int32', padding='post', truncating='pre', value=0.)  #zinc 250,000
        ex_time=time.time()

        for i in range(1):
            global graph
            with graph.as_default():
                predictions=model.predict(x_pad)
                #print "shape of RNN",predictions.shape
                preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
                preds = np.log(preds) / 1.0
                preds = np.exp(preds) / np.sum(np.exp(preds))
                #next_probas = np.random.multinomial(1, preds, 1)
                #print('preds', preds)
                next_probas=np.argsort(preds)[-5:]
                next_probas=list(next_probas)
                #print('next_probas', next_probas)
		#next_int=np.argmax(next_probas)
                #get_int.append(next_int)
                #all_nodes.append(next_int)

        #all_nodes=list(set(all_nodes))
        if 0 in next_probas:
            next_probas.remove(0)
        all_nodes=next_probas
	#print('all_nodes', all_nodes)

        self.expanded=all_nodes
        #print self.expanded
        exfi_time=time.time()-ex_time
	#print exfi_time


    def expanded_node(self, model,state,val):
        all_nodes=[]

        end="\n"
        position=[]
        position.extend(state)
        total_generated=[]
        new_compound=[]
        get_int_old=[]
        for j in range(len(position)):
            get_int_old.append(val.index(position[j]))

        get_int=get_int_old
        x=np.reshape(get_int,(1,len(get_int)))
        x_pad= sequence.pad_sequences(x, maxlen=smiles_max_len, dtype='int32',
            padding='post', truncating='pre', value=0.)
	    #ex_time=time.time()
        for i in range(60):
            global graph
            with graph.as_default():
                predictions=model.predict(x_pad)
                #print "shape of RNN",predictions.shape
                preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
                preds = np.log(preds) / 1.0
                preds = np.exp(preds) / np.sum(np.exp(preds))
                #print('preds', preds)
                next_probas = np.random.multinomial(1, preds, 1)
                next_int=np.argmax(next_probas)
                #print('next_int', next_int)
                #get_int.append(next_int)
                all_nodes.append(next_int)

        all_nodes=list(set(all_nodes))

        #print('all_nodes', all_nodes)
        self.expanded=all_nodes
        #print self.expanded
	#exfi_time=time.time()-ex_time
	#print exfi_time


    def expanded_node_puct(self, model,state,val):
        all_nodes=[]

        end="\n"
        position=[]
        position.extend(state)
        total_generated=[]
        new_compound=[]
        get_int_old=[]
        for j in range(len(position)):
            get_int_old.append(val.index(position[j]))

        get_int=get_int_old
        x=np.reshape(get_int,(1,len(get_int)))
        #x_pad= sequence.pad_sequences(x, maxlen=42, dtype='int32',
        #    padding='post', truncating='pre', value=0.)
        x_pad= sequence.pad_sequences(x, maxlen=smiles_max_len, dtype='int32',padding='post', truncating='pre', value=0.)
        #ex_time=time.time()
        for i in range(1):
            global graph
            with graph.as_default():
                predictions=model.predict(x_pad)
                #print "shape of RNN",predictions.shape
                preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
                preds = np.log(preds) / 1.0
                preds = np.exp(preds) / np.sum(np.exp(preds))
                #print('preds', preds, len(preds))
                next_probas = np.random.multinomial(1, preds, 1)
                next_int=np.argmax(next_probas)
                #print('next_int', next_int)
                #get_int.append(next_int)
                #all_nodes.append(next_int)

        ordered_preds = np.sort(preds)[::-1]
        ordered_index = np.argsort(preds)[::-1]
        #print('ordered_preds', ordered_preds, 'ordered_index', ordered_index)
        cut_index = 0
        p_sum = 0
        for i in range(len(ordered_preds)):
            p_sum += ordered_preds[i]
            #print(i, p_sum)
            if p_sum > 0.99:
                cut_index = i+1
                break
        #all_nodes=list(set(all_nodes))
        all_nodes = ordered_index[:cut_index]
        all_probs = ordered_preds[:cut_index]
        #print('all_nodes', all_nodes, 'all_probs', all_probs)
        self.expanded=all_nodes
        self.all_probs=all_probs



    def node_to_add(self, all_nodes,val):
        added_nodes=[]
        #print('val',val)
        for i in range(len(all_nodes)):
            #print('val[all_nodes[i]]',val[all_nodes[i]],)
            added_nodes.append(val[all_nodes[i]])

        self.nodeadded=added_nodes

        #print "childNodes of current node:", self.nodeadded

    def random_node_to_add(self, all_nodes,val):
        added_nodes=[]
        for i in range(len(all_nodes)):
            added_nodes.append(val[all_nodes[i]])

        #self.random_node=added_nodes





        #print "node.nodeadded:",self.nodeadded





"""Define some functions used for RNN"""



def chem_kn_simulation(model,state,val,added_nodes,mode='mcts'):
    all_posible=[]

    end="\n"

    position=[]
    position.extend(state)
    position.append(added_nodes)
    total_generated=[]
    new_compound=[]
    get_int_old=[]
    for j in range(len(position)):
        get_int_old.append(val.index(position[j]))

    get_int=get_int_old

    x=np.reshape(get_int,(1,len(get_int)))
    x_pad= sequence.pad_sequences(x, maxlen=smiles_max_len, dtype='int32',
        padding='post', truncating='pre', value=0.)
    if mode=='mixed':
        for c in state:
            new_compound.append(c)
        if new_compound in SMILES_historic_list:
            while not get_int[-1] == val.index(end):
                predictions=model.predict(x_pad)
                #print "shape of RNN",predictions.shape
                preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
                preds = np.log(preds) / 1.0
                preds = np.exp(preds) / np.sum(np.exp(preds))
                next_probas = np.random.multinomial(1, preds, 1)
                next_int=np.argmax(next_probas)
                a=predictions[0][len(get_int)-1]
                #next_int_test=sorted(range(len(a)), key=lambda i: a[i])[-10:]
                get_int.append(next_int)
                x=np.reshape(get_int,(1,len(get_int)))
                x_pad = sequence.pad_sequences(x, maxlen=smiles_max_len, dtype='int32',
                    padding='post', truncating='pre', value=0.)
                if len(get_int)>state_length:
                    break
            total_generated.append(get_int)
            all_posible.extend(total_generated)
        else:
            #print('Found New branch')
            SMILES_historic_list.append(new_compound)
            total_generated.append(get_int)
            all_posible.extend(total_generated)
        #print('total_generated:',np.shape(total_generated),total_generated)
        #print('state , m:',state,added_nodes)
    if mode=='mcts':
        while not get_int[-1] == val.index(end):
            predictions=model.predict(x_pad)
            #print "shape of RNN",predictions.shape
            preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
            preds = np.log(preds) / 1.0
            preds = np.exp(preds) / np.sum(np.exp(preds))
            next_probas = np.random.multinomial(1, preds, 1)
            next_int=np.argmax(next_probas)
            a=predictions[0][len(get_int)-1]
            #next_int_test=sorted(range(len(a)), key=lambda i: a[i])[-10:]
            get_int.append(next_int)
            x=np.reshape(get_int,(1,len(get_int)))
            x_pad = sequence.pad_sequences(x, maxlen=smiles_max_len, dtype='int32',
                padding='post', truncating='pre', value=0.)
            if len(get_int)>state_length:
                break
        total_generated.append(get_int)
        all_posible.extend(total_generated)

    return all_posible




def predict_smile(all_posible,val):
    new_compound=[]
    for i in range(len(all_posible)):
        total_generated=all_posible[i]
        generate_smile=[]
        for j in range(len(total_generated)-1):
            generate_smile.append(val[total_generated[j]])
        generate_smile.remove("&")
        new_compound.append(generate_smile)

    return new_compound


def make_input_smile(generate_smile):
    new_compound=[]
    for i in range(len(generate_smile)):
        middle=[]
        for j in range(len(generate_smile[i])):
            middle.append(generate_smile[i][j])
        com=''.join(middle)
        new_compound.append(com)
    return new_compound




def ChemTS_run(rootnode,result_queue,lock,chem_model,ts_strategy,search_parameter,num_simulations, gau_parallel,simulation_time, output_file,alpha,objective,num_rollout,charge_check,SA_score_check,mode):
    """----------------------------------------------------------------------"""
    global maxnum
    global gau_file_index
    global ind_mol
    start_time=time.time()
    while time.time()-start_time<simulation_time:
        node = rootnode
        state=['&']
        """selection step"""
        node_pool=[]
        lock.acquire()

        #print 'node.expanded', node.expanded, 'node.nodeadded', node.nodeadded, 'len(node.childNodes)', len(node.childNodes), len(node.expanded)
        while len(node.expanded)>0 and node.nodeadded==[] and len(node.childNodes)==len(node.expanded):
            #node.num_thread_visited+=1
            #node.virtual_loss+=0
            #print 'state',state,'node.expanded', node.expanded, 'node.nodeadded',node.nodeadded,'len(node.childNodes)',len(node.childNodes), len(node.expanded)

            node = node.Selectnode(ts_strategy, search_parameter, alpha)
            state.append(node.position)
            #print 'state',state,'node.expanded', node.expanded, 'node.nodeadded',node.nodeadded,'len(node.childNodes)',len(node.childNodes), len(node.expanded)
        depth.append(len(state))
	#lock.release()

        """this if condition makes sure the tree not exceed the maximum depth"""
        if len(state)>state_length:
            print('exceed state_length, re = -10')
            re=-10
            while node != None:
                node.Update(re)
                #node.delete_virtual_loss()
                node = node.parentNode
            lock.release()
        else:
            """expansion step"""
            #lock.acquire()
            m = None
            if node.expanded==[]:
                if ts_strategy == 'uct':
                    node.expanded_node(chem_model,state,val)
                elif ts_strategy == 'puct':
                    node.expanded_node_puct(chem_model,state,val)
                node.node_to_add(node.expanded,val)
                node.random_node_to_add(node.expanded,val)

                if node.nodeadded!=[]:

                    m=node.nodeadded[0]
            else:
                if node.nodeadded!=[]:
                    m=node.nodeadded[0]

            if m == None:
                m = val[random.choice(node.expanded)]
                print('randomly selected')
            else:
                if m != '\n':
                    node = node.Addnode(m)
                else:
                    node.nodeadded.remove(m)
                    lock.release()
                    continue
            #print "m is:",m

            lock.release()

	    """simulation step"""
            for ro in range(num_rollout):

                lock.acquire()
                """add virtual loss"""
                node_tmp = node
                while node_tmp != None:
                    #print "node.parentNode:",node.parentNode
                    #node.Update(re)
                    #node.delete_virtual_loss()
                    node_tmp.num_thread_visited+=1
                    node_tmp = node_tmp.parentNode
                ###print 'rootnode.num_thread_visited', rootnode.num_thread_visited
                lock.release()

                lock.acquire()
                maxnum+=1
                ind_mol+=1
                #print('ind_mol', ind_mol)
                #lock.release()
                #"""simulation step"""
                #lock.acquire()
                #print('free_core_id_prev', len(free_core_id),'use_core_id', len(use_core_id))

                dest_core=random.choice(free_core_id)
                use_core_id.append(dest_core)
                free_core_id.remove(dest_core)
                ###print('dest_core', dest_core)
                #generate a new molecule
                for i in range(100):
                    all_posible=chem_kn_simulation(chem_model,state,val,m,mode)
                    generate_smile=predict_smile(all_posible,val)
                    new_compound=make_input_smile(generate_smile)
                    #print type(new_compound),new_compound
            	    #check molecule of duplication
                    if new_compound not in SMILES_historic_list:
                        SMILES_historic_list.append(new_compound)
                        break
                try:
                    comm.send([state,m,ind_mol,new_compound], dest=dest_core, tag=START)
                    lock.release()
                except:
                    print('comm.send failed', Error)
                    free_core_id.append(dest_core)
                    use_core_id.remove(dest_core)
                    lock.acquire()
                    """backpropation step"""
                    while node!= None:
                        #print "node.parentNode:",node.parentNode
                        node.Update(0, add_vis_count = 0)
                        node.delete_virtual_loss()
                        node = node.parentNode
                    lock.release()

                    continue
                #lock.release()

                try:
                    data = comm.recv(source=dest_core, tag=MPI.ANY_TAG, status=status)

                    lock.acquire()
                    free_core_id.append(data[2])
                    use_core_id.remove(data[2])
                    ###print('data[2]', data[2], 'dest_core', dest_core)
                    lock.release()
                except:
                    print('comm.recv failed.')
                    lock.acquire()
                    free_core_id.append(dest_core)
                    use_core_id.remove(dest_core)

                    #data = [-1000, '', 0, 0, 0, 0, 0]
                    """backpropation step"""
                    while node!= None:
                        #print "node.parentNode:",node.parentNode
                        node.Update(0, add_vis_count = 0)
                        node.delete_virtual_loss()
                        node = node.parentNode
                    lock.release()
                    continue

                re = -1
                tag = status.Get_tag()
                if tag == DONE:
                    lock.acquire()
                    all_compounds.append(data[1])
                    lock.release()
                    if objective == 'WL_IT':
                        wl_re = (np.tanh(0.003*(data[0]-400)) + 1)/2
                        intensity_re = (np.tanh((np.log10(data[3]+0.00000001)-np.log10(0.01)))+1)/2
                        w_wl = 0.75
                        w_intensity = 0.25
                        re = w_wl *wl_re+ w_intensity *intensity_re
                    elif objective == 'HL':
                        #HOMO/LUMO
                        re = 1 - data[5]/10.
                    elif objective == 'WL':
                        re = (np.tanh(0.003*(data[0]-400)) + 1)/2
                    elif objective == 'NMR':
                        re = data[13]
                        #print(' re received :', re)


                    #For penality of duplication
                    if data[1] in wave_compounds:
                        print('duplication found, re = -1')
                        re = -1

                    lock.acquire()
                    wave_compounds.append(data[1])
                    wave.append(data[0])
                    deen_list.append(data[4])
                    uv_intensity_list.append(data[3])
                    gap_list.append(data[5])
                    wl_list_list.append(data[6])
                    intensity_list_list.append(data[7])
                    reward_list.append(re)
                    index_list.append(data[8])
                    mol_weight_list.append(data[9])
                    logP_list.append(data[10])
                    SA_score_list.append(data[11])
                    depth_list.append(data[12])
                    nmr_wasser_list.append(data[13])

                    with open('/home/jzhang/code/virtual_loss_wvit_zincSTDwoSsP+-_FP_multiRollout_LC_clear/csvcom_.csv','wb') as file:
                        for line1 in wave_compounds:
                            file.write(str(line1))
                            file.write('\n')
                    with open('/home/jzhang/code/virtual_loss_wvit_zincSTDwoSsP+-_FP_multiRollout_LC_clear/csvwave_.csv','wb') as file:
                        for line2 in wave:
                            file.write(str(line2))
                            file.write('\n')

                    with open('/home/jzhang/'+output_file,'wb') as file:
                        for i in range(len(wave_compounds)):
                            file.write(str(wave_compounds[i])+', ')
                            file.write(str(index_list[i])+', ')
                            file.write(str(mol_weight_list[i])+', ')
                            file.write(str(reward_list[i]))
                            file.write('\n')

                    lock.release()
                    ###if data[0]==-1000:
                        #re=-1
                        ###re=0
                    ###if data[3]<0:
                        #re=-1
                        ###re=0
                    #if m=='\n':
                    #    re=-10000

                lock.acquire()
                #re = re + 1
                if re == None:
                    #print('re is none')
                    re = -1
                """backpropation step"""
                #print('re=', re, data[1], data[8])
                while node!= None:
                    #print "node.parentNode:",node.parentNode
                    if re == -1:
                        re = 0
                        node.Update(re, add_vis_count = 0)
                    else:
                        node.Update(re, add_vis_count = 1)
                    node.delete_virtual_loss()
                    node = node.parentNode
                lock.release()


    result_queue.put([all_compounds,wave_compounds,depth,wave,maxnum,uv_intensity_list,deen_list,gap_list,reward_list,index_list,mol_weight_list,logP_list,SA_score_list,depth_list])

def charge_check(mol):
    print 'charge_checking'
    standard_valence_list = [0, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2]
    check = True
    for atom in mol.GetAtoms():
        if standard_valence_list[atom.GetAtomicNum()] != atom.GetExplicitValence():
            check = False
            break
    return check

def get_hist_list(outdic):
    #print(outdic['nmr'])
    hist_list=[0]*5000
    for i in range(np.shape(outdic['nmr'])[1]):
        if outdic['nmr'][0][i]=='H':
            hist_list[int(outdic['nmr'][1][i]*100)]+=1
    return hist_list

def get_wasser_vect(outdic,element):
    peak_count=0
    peaks=[]
    for i in range(np.shape(outdic['nmr'])[1]):
        if outdic['nmr'][0][i]==element:
            peak_count = peak_count + 1
            peaks.append(outdic['nmr'][1][i])
    peaks_sorted=sorted(peaks)
    peaks_results=[]
    peaks_results.append(peaks_sorted[0])
    peaks_h=[1]
    for i in range(1,len(peaks_sorted)):
        if peaks_sorted[i]==peaks_sorted[i-1]:
            peaks_h[-1] = peaks_h[-1]+1
        else:
            peaks_results.append(peaks_sorted[i])
            peaks_h.append(1)
    peaks_h = [float(i)/peak_count for i in peaks_h]
    peaks_hcumsum = np.cumsum(peaks_h)
    #print(peaks_results,peaks_h,peaks_hcumsum)
    return peaks_results,peaks_h,peaks_hcumsum


def get_wasserstein_dist(outdic, target_outdic):
    peaks_results,peaks_h,peaks_hcumsum = get_wasser_vect(outdic,'H')
    target_peaks_results, target_peaks_h, target_peaks_hcumsum = get_wasser_vect(target_outdic,'H')
    d = wasserstein_distance(peaks_results,target_peaks_results)
    #peak number penality(PNP)
    peaks_number_penality=abs(len(peaks_results)-len(target_peaks_results))
    #end of PNP
    alpha = 0.5
    d = (1 - np.tanh(d)) * (1 - np.tanh(alpha * peaks_number_penality))
    return d

def get_wasserstein_score(hist_list,target):
    score=-wasserstein_distance(target,hist_list)
    #print("score=",score)
    return score

def tree_test_scoring(smiles):
    score = 0
    for i in range(len(smiles)):
        if smiles[i]=='C' or smiles[i]=='c':
            score = score +1
    return score

def gaussion_workers(chem_model,val,gau_parallel,charge_check,output_file_name,lock):
    while True:
        simulation_time=time.time()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        #print('rank_tag:',rank, tag)
        if tag==START:
            state=task[0]
            m=task[1]
            ind=task[2]
            new_compound=task[3]
            print "new compound:",new_compound
            score=[]
            kao=[]
            intensity = -1000000
            deen = 1000000
            gap = 1000000
            mol_weight = 0
            SA_score = 10
            wavenum = -1000
            logP = 0
            dp = len(state)
            nmr_wasser = -1
            intensity_list = []
            wl_list = []
            standard_valence_list = [0, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2]

            try:
                m = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(str(new_compound[0])),True))
                mol_weight = Descriptors.MolWt(m)
                logP = Crippen.MolLogP(m)
                #SA_score = sascorer.calculateScore(m)
                #print 'prev add Hs'
                m_H = Chem.AddHs(m)
                #print Chem.MolToSmiles(m_H)

                standard_valence_list = [0, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2]
                ccheck = True
                if charge_check:
                    for atom in m_H.GetAtoms():
                        #print standard_valence_list[atom.GetAtomicNum()], atom.GetExplicitValence()
                        if standard_valence_list[atom.GetAtomicNum()] != atom.GetExplicitValence():
                            ccheck = False
                            break

                if not ccheck:
                    m = None

            except:
                m=None

            if m!=None:
                H_count=0
                C_count=0
                for atom in m.GetAtoms():
                    H_count=H_count+atom.GetTotalNumHs(includeNeighbors=False)
                    if atom.GetSymbol()=='C':
                        C_count=C_count+1

                if H_count > target_outdic['nmr'][0].count('H') or C_count > target_outdic['nmr'][0].count('C'):
                    nmr_wasser=-1
                    print(str(new_compound[0]),'rejected',H_count,C_count)
                else:
                    print(str(new_compound[0]),'accepeted',H_count,C_count)
                    try:
                        stable=transfersdf(str(new_compound[0]),ind)
                    except:
                        stable = -1
                        print('warning: unstable')
                    if stable==1.0:
                        cd_path = os.getcwd()
                        try:
                            SDFinput = 'CheckMolopt'+str(ind)+'.sdf'
                            calc_sdf = GaussianDFTRun('B3LYP', '3-21G*', gau_parallel, 'nmr', SDFinput, 0)
                            outdic = calc_sdf.run_gaussian()
                            nmr_wasser=get_wasserstein_dist(outdic,target_outdic)
                            print ind,'|',nmr_wasser,'|',new_compound[0],'|',outdic['nmr']
                            if os.path.isfile('CheckMol'+str(ind)+'.sdf'):
                                shutil.move('CheckMol'+str(ind)+'.', 'dft_result')
                            if os.path.isfile('CheckMolopt'+str(ind)+'.sdf'):
                                shutil.move('CheckMolopt'+str(ind)+'.sdf', 'dft_result')
                        except:
                            os.chdir(cd_path)
                            if os.path.isfile('CheckMolopt'+str(ind)+'.sdf'):
                                os.remove('CheckMolopt'+str(ind)+'.sdf')
                            if os.path.isfile('CheckMol'+str(ind)+'.sdf'):
                                os.remove('CheckMol'+str(ind)+'.sdf')
                    else:
                        wavelength=None
                        if os.path.isfile('CheckMolopt'+str(ind)+'.sdf'):
                            os.remove('CheckMolopt'+str(ind)+'.sdf')
                        if os.path.isfile('CheckMol'+str(ind)+'.sdf'):
                            os.remove('CheckMol'+str(ind)+'.sdf')
            score.append(wavenum)
            score.append(new_compound[0])
            score.append(rank)
            score.append(intensity)
            score.append(deen)
            score.append(gap)
            score.append(wl_list)
            score.append(intensity_list)
            score.append(ind)
            score.append(mol_weight)
            score.append(logP)
            score.append(SA_score)
            score.append(dp)
            score.append(nmr_wasser)

            comm.send(score, dest=0, tag=DONE)
            simulation_fi_time=time.time()-simulation_time
            ###print("simulation_fi_time:",simulation_fi_time)
        if tag==EXIT:
            MPI.Abort(MPI.COMM_WORLD)

    comm.send([-1000,'',0,0,0,0,[],[],ind,0,0,0,0], dest=0, tag=EXIT)

def new_read_database(file_name):
    smiles_list=[]
    index_list=[]
    nmr_list=[]
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter='|')
        for row in csv_reader:
            smiles_list.append(row[0])
            index_list.append(int(row[1]))
            nmr_list.append(ast.literal_eval(row[2]))
        #print(row[0],int(row[1]),type(ast.literal_eval(row[2])))
    return smiles_list,index_list,nmr_list

SMILES_historic_list=[]
cnf = ConfigParser.ConfigParser()
cnf.read('set.ini')
target_filename=cnf.get('target', 'target_filename')
target_index=int(cnf.get('target','target_index'))
target_list,target_index_list,target_nmr_list=new_read_database(target_filename)
target_outdic={'nmr':target_nmr_list[target_index]}
target_mol=target_list[target_index]

if __name__ == "__main__":
    comm=MPI.COMM_WORLD
    size=comm.size
    rank=comm.rank
    status=MPI.Status()
    READY, START, DONE, EXIT = 0, 1, 2, 3
    val=['\n', '&', 'O', 'c', '1', '(', ')', '=', 'C', 'N', '#', 'n', '2', 'o', '3', '-', '4']
    chem_model=loaded_model()
    graph = tf.get_default_graph()
    chemical_state = chemical()
    ts_strategy = cnf.get('param','ts_strategy') #'uct', 'puct'
    search_parameter = float(cnf.get('param','search_parameter')) #If ts_strategy=='uct', 0 < search_parameter < 1. If ts_strategy=='puct', default value is 5 (AlphaGo).
    num_simulations = int(cnf.get('param','num_simulations'))  # core - 1, max: 2560 (skylake)
    gau_parallel = 1
    num_rollout = int(cnf.get('param','num_rollout'))
    simulation_time = 3600*int(cnf.get('param','simulation_time')) # 3600*24 # max: 168h
    trie = int(cnf.get('param','trie'))
    random_trie = int(cnf.get('param','random_trie'))
    alpha = 1 # alph*mean + (1 - alpha)*max + bais
    objective = 'NMR' # 'WL_IT', 'HL', 'WL', 'NMR'
    charge_check = False # True or False
    SA_score_check = False # True or False
    mode=cnf.get('param','mode')#mixed or mcts
    output_file = 'result10k_'+ts_strategy+'_'+str(mode)+'_C'+str(search_parameter)+'_alpha'+str(alpha)+'_para'+str(num_simulations)+'_time'+str(simulation_time/3600)+'h_rollout'+str(num_rollout)+'_'+target_mol+'_trie_'+str(trie)+'_randtrie_'+str(random_trie)+'_'+str(target_index)+'_'+str(time.strftime("%Y%m%d-%H%M%S"))+'.csv'
    print num_simulations,gau_parallel,num_rollout,simulation_time,search_parameter
    #h = hpy()
    #print h.heap()
    thread_pool=[]
    lock=Lock()
    gau_file_index=0

    """initialization of the chemical trees and grammar trees"""
    root=['&']
    rootnode = Node(position= root)
    maxnum=0
    ind_mol=0
    reward_dis=[]
    all_compounds=[]
    wave_compounds=[]
    wave=[]
    deen_list = []
    gap_list = []
    uv_intensity_list = []
    wl_list_list = []
    intensity_list_list = []
    reward_list = []
    index_list = []
    mol_weight_list = []
    logP_list = []
    SA_score_list = []
    depth_list = []
    nmr_wasser_list = []
    depth=[]
    result=[]
    result_queue=Queue()
    free_core_id=range(1,num_simulations+1)
    use_core_id = []
    if trie != 0:
        #free_core_id=[40,80,120,160,200,240,280,320,360,400,440,480,520,560,600,640,680,720,760,800,840,880,920,960,1000]
        #DB Input
        smiles_list,index_list,nmr_list = new_read_database('2019_v3_noCH.smi')
        #print nmr_list
        #Trie Build
        #rootnode.expanded_node_puct(chem_model,['&'],val)
        store_list=[]
        #print type(target_outdic['nmr']),type(nmr_list[0])
        if random_trie == 1:
            smiles_list=smiles_list[trie]
        else:
            for j in range(len(nmr_list)):
                store_list.append([get_wasserstein_dist({'nmr':nmr_list[j]},target_outdic),smiles_list[j]])
            store_list= sorted(store_list,reverse=True, key=lambda x: (x[0],x[1]))[:trie]
            #print store_list
            #print np.shape(store_list)
            smiles_list= [row[1] for row in store_list]
        for i in range(len(smiles_list)):
            #print i
            j=0
            state=['&']
            current_node=rootnode
            while j <= len(smiles_list[i]) - 1 :
                state.append(smiles_list[i][j])
                #current_node.expanded_node_puct(chem_model,state,val)
                #print current_node.expanded
                if val.index(smiles_list[i][j]) not in current_node.expanded:
                    new_node = Node(position= smiles_list[i][j])
                    #new_node.expanded_node_puct(chem_model,state,val)
                    #np.append(current_node.expanded,val.index(smiles_list[i][j]))
                    #np.append(current_node.all_probs,0.001)
                    current_node.childNodes.append(new_node)
                    new_node.parentNode = current_node
                    current_node=new_node
                else:
                    for child in current_node.childNodes:
                        if child.position == smiles_list[i][j]:
                            current_node=child
                            break
                j = j + 1
            current_nmr={}
            current_nmr['nmr']=nmr_list[i]
            #print current_nmr
            nmr_wasser=get_wasserstein_dist(current_nmr,target_outdic)
            #print i,nmr_wasser
            while current_node != None:
                current_node.Update(nmr_wasser)
                #all_probs=[1/len(current_node.expanded)]*len(current_node.expanded)
                #all_probs=[0.001]*len(current_node.expanded)
                #current_node.all_probs=all_probs
                current_node = current_node.parentNode

    if rank==0:
        for thread_id in range(num_simulations):
            thread_best = Thread(target=ChemTS_run,args=(rootnode,result_queue,lock,chem_model,ts_strategy,search_parameter,num_simulations, gau_parallel,simulation_time,output_file,alpha,objective,num_rollout,charge_check,SA_score_check,mode))
            thread_pool.append(thread_best)

        for i in range(num_simulations):
            thread_pool[i].start()

        for i in range(num_simulations):
            thread_pool[i].join()
        for i in range(num_simulations):
            result.append(result_queue.get())

        comm.Abort()
        for i in range(len(free_core_id)):
            comm.send(None, dest=i+1, tag=EXIT)
        #h = hpy()
        #print h.heap()
    #elif rank%40==0 and rank!=0:
    else:
        #h = hpy()
        #print h.heap()
        gaussion_workers(chem_model,val,gau_parallel, charge_check,output_file,lock)
