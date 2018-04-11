# -*- coding: utf-8 -*-
"""
Criado na Terça-feira de 27 de Março de 2018 às 19:06:14
Última edição em 10/04/2018 às 22:40

@author: ratof
"""
#%%
import csv
import numpy as np
import time
from os import listdir
#%%
# Definição de funções usadas no código todas abaixo

# Potência ativa
def pota(barra,theta,vpu,gpu,bpu,pos_viz_K): # x é pos 
    p = 0                                                        
    for i in pos_viz_K[barra]:                           
        p = p + \
        vpu[barra]*vpu[i]*(gpu[barra,i]*np.cos(theta[barra]-theta[i])+\
           bpu[barra,i]*np.sin(theta[barra]-theta[i]))
    return p
# Potência reativa
def potr(barra,theta,vpu,gpu,bpu,pos_viz_K): # x é pos
    q = 0
    for i in pos_viz_K[barra]:
        q = q + \
        vpu[barra]*vpu[i]*(gpu[barra,i]*np.sin(theta[barra]-theta[i])-\
           bpu[barra,i]*np.cos(theta[barra]-theta[i]))
    return q
# Matriz H
def mat_h(h,theta,vpu,gpu,bpu,pos_viz_K,pos_nvtheta):
    lin,col = h.shape
    for i in range(lin):
        for d in range(col):
            if i != d:
                h[i,d] = vpu[i]*vpu[d]*(gpu[i,d]*np.sin(theta[i]-theta[d])\
                 -bpu[i,d]*np.cos(theta[i]-theta[d]))
            else:
                h[i,d] = -1*vpu[i]*vpu[i]*bpu[i,d]
                for k in pos_viz_K[i]:
                    h[i,d] = h[i,d] - vpu[i]*vpu[k]*(gpu[i,k]*np.sin(theta[i]\
                     -theta[k])-bpu[i,k]*np.cos(theta[i]-theta[k]))
    for i in range(lin):
        for d in range(col):
            if i == pos_nvtheta and d == pos_nvtheta:
                h[i,d] = np.inf
    return h
# Matriz N
def mat_n(n,theta,vpu,gpu,bpu,pos_viz_K):
    lin,col = n.shape
    for i in range(lin):
        for d in range(col):
            if i != d:
                n[i,d] = vpu[i]*(gpu[i,d]*np.cos(theta[i]-theta[d])+bpu[i,d]*\
                 np.sin(theta[i]-theta[d]))
            else:
                n[i,d] = vpu[i]*gpu[i,d]
                for k in pos_viz_K[i]:
                    n[i,d] = n[i,d] + vpu[k]*(gpu[i,k]*np.cos(theta[i]-\
                     theta[k])+bpu[i,k]*np.sin(theta[i]-theta[k]))
    return n
# Matriz M
def mat_m(m,theta,vpu,gpu,bpu,pos_viz_K):
    lin,col = m.shape
    for i in range(lin):
        for d in range(col):
            if i != d:
                m[i,d] = -1*vpu[i]*vpu[d]*(gpu[i,d]*np.cos(theta[i]-theta[d])+\
                 bpu[i,d]*np.sin(theta[i]-theta[d]))
            else:
                m[i,d] = -1*vpu[i]*vpu[i]*gpu[i,d]
                for k in pos_viz_K[i]:
                    m[i,d] = m[i,d] + vpu[i]*vpu[k]*(gpu[i,k]*np.cos(theta[i]-\
                     theta[k])+bpu[i,k]*np.sin(theta[i]-theta[k]))
    return m
# Matriz L
def mat_l(l,theta,vpu,gpu,bpu,pos_viz_K,pos_nvtheta,pos_npv):
    lin,col = l.shape
    for i in range(lin):
        for d in range(col):
            if i != d:
                l[i,d] = vpu[i]*(gpu[i,d]*np.sin(theta[i]-theta[d])-bpu[i,d]*\
                 np.cos(theta[i]-theta[d]))
            else:
                l[i,d] = -1*vpu[i]*bpu[i,d]
                for k in pos_viz_K[i]:
                    l[i,d] = l[i,d] + vpu[k]*(gpu[i,k]*np.sin(theta[i]-\
                     theta[k])-bpu[i,k]*np.cos(theta[i]-theta[k]))
    for i in range(lin):
        for d in range(col):
            if i == pos_nvtheta and d == pos_nvtheta:
                l[i,d] = np.inf
    for i in range(lin):
        for d in range(col):
            if i in pos_npv and d in pos_npv:
                l[i,i] = np.inf
    return l
# Menu para impressão
def menu():
    print('\nConcluído.\n\nEscolha uma entre as opções abaixo para'+\
      ' impressão dos resultados ou entre com qualquer outro valor para '+\
      'sair:\n\n1 - Perdas ativas na transmissão\n2 - Perdas reativas'+\
      ' na transmissão\n3 - Tensões e ângulos finais\n4 - Potências ativas'+\
      ' líquidas injetadas nas barras\n5 - Potências reativas líquidas'+\
      ' injetadas nas barras\n6 - Fluxo de potência ativa na transmissão\n'+\
      '7 - Fluxo de potência reativa na transmissão')
#%%
# Ler arquivo txt de origem e transformar linhas do arquivo em dados de lista
def programa():
    arq_txt = [x for x in listdir() if x.endswith('.txt')]
    escolha = []

    for i in range(len(arq_txt)):
        escolha.append(str(i+1)+" - "+arq_txt[i])
        
    print("Escolha o arquivo da lista abaixo e entre com o valor inteiro:\t")
    print("Obs.: O arquivo deve estar na mesma pasta que o script\n\n")
    for i in range(len(escolha)):
        print(escolha[i])
    print('\nEscolha: ')
    
    escolhas_possiveis = []
    for i in list(range(1,len(arq_txt)+1)):
        escolhas_possiveis.append(str(i))
        
    escolhido = input()
    
    while escolhido not in escolhas_possiveis:
        escolhido = input()
    
    print('\nDigite o Erro Ep e Eq ou tecle Enter para o erro padrão (10e-5):\n')
    e = input()
    if e == "":
        e = 0.00001
    else:
        e = float(e)    
    
    tempo_inicio = time.time()
            
    with open(arq_txt[int(escolhido)-1]) as dados_origem_txt:
        leitura = csv.reader(dados_origem_txt)
        dados = list(leitura)
        
    # Encontrando a posição dos fins da lista de dados, tomando como base o
    # indicador '-999'
        
    cortes=[] #Iniciando lista que conterá as posições dos '-999'
    
    for i in range(len(dados)):
        if dados[i-1][0][:4] == '-999':#1º [] indica linha,2º [] sublista única
            cortes.append(i-1)         #e 3º [] é o corte de 4 caracteres
                                           #Se os 4 primeiros caracteres são iguais
                                           #a '-999', o índice é computado na lista
                                           #cortes
    
    del(cortes[-1]) #Eliminando o último elemento porque não tem utilidade
    
    sbase = float(dados[0][0][31:36])
    
    # Nota: Os dados de nós começam na 3ª linha e acabam em cortes[0]-1 e os dados
    # de ramos começam em cortes[0]+2 e acabam em cortes[1]-1. Além disso, a 2ª
    # coluna dos dados de nós contém strings
    
    #Elimando a segunda coluna (strings) dos dados de nós para realizar a conversão
    # para arrays
    
    for i in range(2,cortes[0]):
        dados[i][0]=dados[i][0][:5]+dados[i][0][17:]
    #%%        
    #Criando lista de nós
        
    nos = []
    nb  = 0
    for i in range(2,cortes[0]):
        nos.append(dados[i])
        nb += 1
    #%%    
    #Criando lista de ramos
        
    nr = 0
    ramos = []
    for i in range(cortes[0]+2,cortes[1]):
        ramos.append(dados[i])
        nr += 1
    #%%    
        
    # Usando NumPy para criar array de nós
        
    nos_dados = np.arange(nb*17,dtype=float).reshape(nb,17)
    for i in range(nb):
        nos_dados[i] = np.fromstring(nos[i][0],dtype=float,sep=' ')
        
    # O respectivo cabeçalho para coluna é (17 colunas): 
    """  
    1   Bus number (I) *
    2   Load flow area number (I) Don't use zero! *
    3   Loss zone number (I)
    4   Type (I) *
            0 - Unregulated (load, PQ)
            1 - Hold MVAR generation within voltage limits, (PQ)
            2 - Hold voltage within VAR limits (gen, PV)
            3 - Hold voltage and angle (swing, V-Theta) (must always have one)
    5   Final voltage, p.u. (F) *
    6   Final angle, degrees (F) *
    7   Load MW (F) *
    8   Load MVAR (F) *
    9   Generation MW (F) *
    10  Generation MVAR (F) *
    11  Base KV (F)
    12  Desired volts (pu) (F) (This is desired remote voltage if this bus is 
        controlling another bus.
    13  Maximum MVAR or voltage limit (F)
    14  Minimum MVAR or voltage limit (F)
    15  Shunt conductance G (per unit) (F) *
    16  Shunt susceptance B (per unit) (F) *
    17  Remote controlled bus number
    """
    #%%    
    # Usando NumPy para criar array de ramos
        
    ramos_dados = np.arange(nr*21,dtype=float).reshape(nr,21)
    for i in range(nr):
        ramos_dados[i] = np.fromstring(ramos[i][0],dtype=float,sep=' ')
        
    # A partir deste ponto, os dados estão prontos para serem trabalhados
    # O respectivo cabeçalho para cada coluna é (21 colunas):
    """
    1   Tap bus number (I) *
                     For transformers or phase shifters, the side of the model
                     the non-unity tap is on
    2   Z bus number (I) *
                     For transformers and phase shifters, the side of the model
                     the device impedance is on.
    3   Load flow area (I)
    4   Loss zone (I)
    5   Circuit (I) * (Use 1 for single lines)
    6   Type (I) *
                     0 - Transmission line
                     1 - Fixed tap
                     2 - Variable tap for voltage control (TCUL, LTC)
                     3 - Variable tap (turns ratio) for MVAR control
                     4 - Variable phase angle for MW control (phase shifter)
    7   Branch resistance R, per unit (F) *
    8   Branch reactance X, per unit (F) * No zero impedance lines
    9   Line charging B, per unit (F) * (total line charging, +B)
    10  Line MVA rating No 1 (I) Left justify!
    11  Line MVA rating No 2 (I) Left justify!
    12  Line MVA rating No 3 (I) Left justify!
    13  Control bus number
    14  Side (I)
                     0 - Controlled bus is one of the terminals
                     1 - Controlled bus is near the tap side
                     2 - Controlled bus is near the impedance side (Z bus)
    15  Transformer final turns ratio (F)
    16  Transformer (phase shifter) final angle (F)
    17  Minimum tap or phase shift (F)
    18  Maximum tap or phase shift (F)
    19  Step size (F)
    20  Minimum voltage, MVAR or MW limit (F)
    21  Maximum voltage, MVAR or MW limit (F)
    """
    #%%
    # Criar lista com vizinhos e outra com posições de cada barra, excluindo-se a 
    # própria barra. A sintaxe é, por exemplo, vizinhos da barra k -> vizinhos[k-1]
    # Sintaxe de posições é semelhante, posições de vizinhos de k (em ramos_dados):
    # é -> posicoes_vizinhos[k-1]
    
    vizinhos = []            # Vizinhos não será usada, apenas para consulta
    vizinhos_K = []
    posicoes_vizinhos = []   # Onde serão alocadas as posições das barras vizinhas
    pos_viz_K = []
    posicoes_vizinhos_ramos = [] # lista com posição de linha de cada vizinho 
                                 # NOS DADOS DE RAMOS!!!
                             
    for i in range(len(nos_dados)):
        vizinhos.append([])
        vizinhos_K.append([])
        posicoes_vizinhos.append([])
        pos_viz_K.append([])
        posicoes_vizinhos_ramos.append([])
        for j in range(len(ramos_dados)):
            if int(nos_dados[i][0]) == int(ramos_dados[j][0]):
                posicoes_vizinhos_ramos[i].append(j)
                vizinhos[i].append(int(ramos_dados[j][1]))
                vizinhos_K[i].append(int(ramos_dados[j][1]))
                posicoes_vizinhos[i].append(nos_dados[:,0].tolist().\
                                 index(ramos_dados[j][1]))
                pos_viz_K[i].append(nos_dados[:,0].tolist().\
                                   index(ramos_dados[j][1]))
            elif int(nos_dados[i][0]) == int(ramos_dados[j][1]) and \
            int(ramos_dados[j][0])\
            not in vizinhos[i]:
                posicoes_vizinhos_ramos[i].append(j)
                vizinhos[i].append(int(ramos_dados[j][0]))
                vizinhos_K[(i)].append(int(ramos_dados[j][0]))
                posicoes_vizinhos[i].append(nos_dados[:,0].tolist().\
                                 index(ramos_dados[j][0])) 
                pos_viz_K[i].append(nos_dados[:,0].tolist().\
                                   index(ramos_dados[j][0]))
        vizinhos_K[i].append(i+1)
        pos_viz_K[i].append(nos_dados[:,0].tolist().\
                           index(nos_dados[:,0][i]))
    
    for i in vizinhos:                 # Cada vizinho, vizinhos incluindo a
        i.sort()                       # própria barra, posições dos vizinhos
    for i in vizinhos_K:             # e posições dos vizinhos incluindo a
        i.sort()                     # a própria barra são ordenados em
    for i in posicoes_vizinhos:        # ordem crescente
        i.sort()
    for i in pos_viz_K:
        i.sort()  
    #%%
    pos_nos_em_nos = nos_dados[:,0]
    pos_nos_em_nos = pos_nos_em_nos.tolist()
    
    pos_nos_em_ramos = ramos_dados[:,0::]
    pos_nos_em_ramos = pos_nos_em_ramos.tolist()
    
    for i in range(len(pos_nos_em_ramos)):
        for term in range(2):
            pos_nos_em_ramos[i][term] = float(pos_nos_em_nos.\
                            index(pos_nos_em_ramos[i]\
                            [term]))
            
    #%%
    # Iniciando Mat de Admitância. Lembrando que Y = G + jB e a dimensão é nb x nb
    
    ypu = np.arange(nb*nb).reshape(nb,nb).astype('complex')
    ypu.fill(0)  # Preeche todos elementos com 0
    #%%
    # Montando diagonal principal, sempre não nula
    
    for i in range(nb):
        ypu[i][i] = nos_dados[i,14]+nos_dados[i,15]*1j
        for j in posicoes_vizinhos_ramos[i]:
            ypu[i][i] = ypu[i][i] + ramos_dados[j][8]*(1j/2) + \
            (1/(ramos_dados[j][7]*1j + ramos_dados[j][6]))
    
    # Montando elementos fora da diagonal principal
    # Nota: onde não há ramo conectando barras o loop nem verifica, permanecendo 0
            
    for i in ramos_dados:
        ypu[int(i[0])-1,int(i[1])-1]=-1*(1/(i[7]*1j+i[6])*\
        np.exp(-1*(i[15]*1j)))
        ypu[int(i[1])-1,int(i[0])-1]=-1*(1/(i[7]*1j+i[6])*\
        np.exp(1*(i[15]*1j)))
    
    gpu = np.real(ypu) # Parte real de Ypu
    bpu = np.imag(ypu) # Parte imaginária de Ypu
    #%%
    # Calculando número de equações NPQ e NPV para dimensão da Jacobiana
    
    #        H      N   } NPQ+NPV
    #        M      L   } NPQ
    #       \/     \/
    #     NPQ+NPV  NPQ
    
    npq,npv,nvtheta = (0,0,0)
    pos_npq = []
    pos_npv = []
    
    
    for i in range(len(nos_dados[:,3])):
        if nos_dados[:,3][i] == 0.0 or nos_dados[:,3][i] == 1.0:
            npq = npq + 1
            pos_npq.append(i)
        elif nos_dados[:,3][i] == 2.0:
            npv = npv + 1
            pos_npv.append(i)
        else:
            nvtheta = nvtheta + 1
            pos_nvtheta = i
    
    
    #%%        
    # Iniciando as submatrizes da Jacobiana
            
    h = np.arange(nb*nb,dtype=np.float64).reshape(nb,nb)
    h.fill(0)
    n = np.arange(nb*nb,dtype=np.float64).reshape(nb,nb)
    n.fill(0)
    m = np.arange(nb*nb,dtype=np.float64).reshape(nb,nb)
    m.fill(0)
    l = np.arange(nb*nb,dtype=np.float64).reshape(nb,nb)
    l.fill(0)
    
    #%%
    jac = np.arange((2*nb)*(2*nb),dtype=np.float64).reshape(2*nb,2*nb)
    jac.fill(0)
    
    #%%
    # Dados e Incognitas
    
    nos_pos = []
    
    for i in range(len(nos_dados[:,0])):
        nos_pos.append(i)
        
    for i in range(len(nos_dados[:,0])):
        nos_pos.append(i)
    
    ppu = (nos_dados[:,8]-nos_dados[:,6])/sbase
    qpu = (nos_dados[:,9]-nos_dados[:,7])/sbase
    vpu = nos_dados[:,4]
    theta = nos_dados[:,5]*(np.pi/180.) # vetor theta convertido em radianos
    
    #%%
    # Arrumando vetores, ou seja, para barras PQ, V e theta são incógnitas. Para
    # barras PV, Q e theta são incógnitas. Para barras Vtheta, P e Q são incógnitas
    # Os chutes iniciais são baseados na barra de referência e já são ajustados
    # abaixo
    
    ref_vpu = nos_dados[pos_nvtheta,4]
    ref_theta = nos_dados[pos_nvtheta,5]
    
    for i in pos_npq:
        vpu[i] = ref_vpu
        theta[i] = ref_theta
            
    for i in pos_npv:
        theta[i] = ref_theta
        
    #%%
    vetor_pos_pa = list(pos_npq)
    for i in pos_npv:
        vetor_pos_pa.append(i)
    vetor_pos_pa.sort()
    #%%
    vetor_pos_pr = list(pos_npq)
    vetor_pos_pr.sort()
    #%%
    vetor_pos_dados = list(vetor_pos_pa)
    for i in vetor_pos_pr:
        vetor_pos_dados.append(i)
    #%%
    corte_theta = len(vetor_pos_pa)
    
    vetor_np_pos_pa = np.array(vetor_pos_pa).reshape(len(vetor_pos_pa),1)
    vetor_np_pos_pr = np.array(vetor_pos_pr).reshape(len(vetor_pos_pr),1)
    vetor_np_pos_pa = vetor_np_pos_pa.astype(dtype=np.float64)
    vetor_np_pos_pr = vetor_np_pos_pr.astype(dtype=np.float64)
    
    esp_p = list(ppu)
    
    esp_q = list(qpu)
    
    esp = list(esp_p)
    for i in esp_q:
        esp.append(i)
    
    deltap = np.copy(vetor_np_pos_pa)
    deltaq = np.copy(vetor_np_pos_pr)
    
    for barra in range(len(deltap)):
        deltap[barra][0] = e + 1
    for barra in range(len(deltaq)):
        deltaq[barra][0] = e + 1
    #%%
    delta = np.arange(2*nb).reshape(2*nb,1)
    delta = delta.astype(dtype=np.float64)
    del_e = np.copy(delta)
    for i in range(len(delta)):
        delta[i][0] = e + 1
    
    #%%    
    iteracao = 0 
    #%%
    #-----------------------------SUBSISTEMA I-------------------------------------
    while (abs(del_e).max()) >= e:
    
        for i in range(2*nb):
            if i < nb:
                delta[i][0] = esp[i] - pota(nos_pos[i],theta,vpu,gpu,bpu,pos_viz_K)
            else:
                delta[i][0] = esp[i] - potr(nos_pos[i],theta,vpu,gpu,bpu,pos_viz_K)
                
        h.fill(0)
        n.fill(0)
        m.fill(0)
        l.fill(0)
        
        mat_h(h,theta,vpu,gpu,bpu,pos_viz_K,pos_nvtheta)
        mat_n(n,theta,vpu,gpu,bpu,pos_viz_K)
        mat_m(m,theta,vpu,gpu,bpu,pos_viz_K)
        mat_l(l,theta,vpu,gpu,bpu,pos_viz_K,pos_nvtheta,pos_npv)
        
    # Incluindo dados das submatrizes em -J
        
        lin,col = h.shape
        for i in range(lin):
            for d in range(col):
                jac[i,d] = h[i,d]
            
        lin,col = n.shape
        for i in range(lin):
            for j in range(col):
                jac[i,d+nb] = n[i,d]
        
        lin,col = m.shape
        for i in range(lin):
            for d in range(col):
                jac[i+nb,d] = m[i,d]
        
        lin,col = l.shape
        for i in range(lin):
            for d in range(col):
                jac[i+nb,d+nb] = l[i,d]
            
        jacinv = np.linalg.inv(jac)
        
        sol = jacinv@delta
        
        for i in range(nb):
            theta[i] = theta[i] + sol[i][0]
            vpu[i] = vpu[i] + sol[i+nb][0]
        
        for i in range(2*nb):
            if i not in vetor_pos_dados:
                del_e[i][0] = 0
            else:
                del_e[i][0] = delta[i][0]
                
        iteracao = iteracao + 1
    #%%
    #----------------------------SUBSISTEMA II-------------------------------------
    # Potência na barra de referência
    for i in pos_npv:
        ppu[i] = pota(i,theta,vpu,gpu,bpu,pos_viz_K)
    
    ppu[pos_nvtheta] = pota(pos_nvtheta,theta,vpu,gpu,bpu,pos_viz_K)
    qpu[pos_nvtheta] = potr(pos_nvtheta,theta,vpu,gpu,bpu,pos_viz_K)
    #------------------------------------------------------------------------------
    #%%
    #---------------------------SUBSISTEMA III-------------------------------------
    # Fluxos de potência
    pfluxo = np.arange(nb*nb,dtype=np.float64).reshape(nb,nb)
    pfluxo.fill(0)
    
    for i in pos_nos_em_ramos:
        conv = (1/((i[6]+i[7]*1j)))
        
        pfluxo[int(i[0]),int(i[1])] = np.real(conv)*(vpu[int(i[0])]**2)-\
        vpu[int(i[0])]*vpu[int(i[1])]*(np.real(conv)*np.cos(theta[int(i[0])]-\
           theta[int(i[1])])+np.imag(conv)*np.sin(theta[int(i[0])]-\
                 theta[int(i[1])]))
        
        pfluxo[int(i[1]),int(i[0])] = np.real(conv)*(vpu[int(i[1])]**2)-\
        vpu[int(i[0])]*vpu[int(i[1])]*(np.real(conv)*np.cos(theta[int(i[0])]-\
           theta[int(i[1])])-np.imag(conv)*np.sin(theta[int(i[0])]-\
                 theta[int(i[1])]))
    
    # Sugestão: nos problemas pedidos não existe transformador defasador, então
    # phi_km é nulo, porém pode-se colocá-lo no cálculo
    
    qfluxo = np.arange(nb*nb,dtype=np.float64).reshape(nb,nb)
    qfluxo.fill(0)
    
    for i in pos_nos_em_ramos:
        conv = (1/((i[6]+i[7]*1j)))
        
        qfluxo[int(i[0]),int(i[1])] = -1*(np.imag(conv)+(i[8]/2))*\
        (vpu[int(i[0])]**2)-vpu[int(i[0])]*vpu[int(i[1])]*(np.real(conv)*\
             np.sin(theta[int(i[0])]-theta[int(i[1])])-np.imag(conv)*\
             np.cos(theta[int(i[0])]-theta[int(i[1])]))
        
        qfluxo[int(i[1]),int(i[0])] = -1*(np.imag(conv)+(i[8]/2))*\
        (vpu[int(i[1])]**2)+vpu[int(i[0])]*vpu[int(i[1])]*(np.real(conv)*\
             np.sin(theta[int(i[0])]-theta[int(i[1])])+np.imag(conv)*\
             np.cos(theta[int(i[0])]-theta[int(i[1])]))
        
    print('\nO sistema convergiu na '+str(iteracao)+\
          'ª iteração\nO tempo de execução foi de '+str(time.time()-tempo_inicio)+\
          ' segundos\n\nAperte Enter para continuar\n')
    input()
    
    #------------------------------------------------------------------------------
    #%%
    # Perdas e resultados do sistema
    
    perdas_ativas = []
    perdas_reativas = []
    tensoes_finais_e_angulos = []
    potencias_ativas = []
    potencias_reativas = []
    fluxo_ativo = []
    fluxo_reativo = []
    
    for i in pos_nos_em_ramos:
        perdas_ativas.append(['Perdas ativas entre barra '+\
                              str(int(pos_nos_em_nos[int(i[0])]))+' e '+\
                              str(int(pos_nos_em_nos[int(i[1])]))+' (ou '+\
                              str(int(pos_nos_em_nos[int(i[1])]))+' e '+\
                              str(int(pos_nos_em_nos[int(i[0])]))+'): '+str\
                              (np.round(pfluxo[int(i[0]),int(i[1])]+\
                                      pfluxo[int(i[1]),int(i[0])],5))+' pu'])
        
        perdas_reativas.append(['Perdas reativas entre barra '+\
                              str(int(pos_nos_em_nos[int(i[0])]))+' e '+\
                              str(int(pos_nos_em_nos[int(i[1])]))+' (ou '+\
                              str(int(pos_nos_em_nos[int(i[1])]))+' e '+\
                              str(int(pos_nos_em_nos[int(i[0])]))+'): '+str\
                              (np.round(qfluxo[int(i[0]),int(i[1])]+\
                                      qfluxo[int(i[1]),int(i[0])],5))+' pu'])
                             
        fluxo_ativo.append(['Fluxo de potência ativa da barra '+\
                              str(int(pos_nos_em_nos[int(i[0])]))+' para a '+\
                              str(int(pos_nos_em_nos[int(i[1])]))+' e '+\
                              str(int(pos_nos_em_nos[int(i[1])]))+' para '+\
                              str(int(pos_nos_em_nos[int(i[0])]))+\
                              ', respectivamente: '+str\
                              (np.round(pfluxo[int(i[0]),int(i[1])],5))+' e '+\
                                      str(np.round(pfluxo[int(i[1]),int(i[0])],5))\
                                      +' pu'])
        fluxo_reativo.append(['Fluxo de potência reativa da barra '+\
                              str(int(pos_nos_em_nos[int(i[0])]))+' para a '+\
                              str(int(pos_nos_em_nos[int(i[1])]))+' e '+\
                              str(int(pos_nos_em_nos[int(i[1])]))+' para '+\
                              str(int(pos_nos_em_nos[int(i[0])]))+\
                              ', respectivamente: '+str\
                              (np.round(qfluxo[int(i[0]),int(i[1])],5))+' e '+\
                                      str(np.round(qfluxo[int(i[1]),int(i[0])],5))\
                                      +' pu'])
    for i in range(nb):
        tensoes_finais_e_angulos.append(['A tensão final da barra '+\
                               str(int(pos_nos_em_nos[i]))+' é '+\
                               str(np.round(vpu[i],5))+\
                                  ' pu, com ângulo '+str(np.round(theta[i],5))+\
                                  ' radianos (ou '+\
                                  str(np.round(np.degrees(theta[i]),5))+\
                                  '°) em relação à referência'])
        
        potencias_ativas.append(['Potência ativa líquida injetada na barra '+\
                                 str(int(pos_nos_em_nos[i]))+' é '+\
                                 str(np.round(pota(nos_pos[i],\
                                          theta,vpu,gpu,bpu,pos_viz_K),5))+' pu'])
        
        potencias_reativas.append(['Potência reativa líquida injetada na barra '+\
                                 str(int(pos_nos_em_nos[i]))+' é '+\
                                 str(np.round(potr(nos_pos[i],\
                                          theta,vpu,gpu,bpu,pos_viz_K),5))+' pu'])
    #%%
    # Impressão
    
    menu()    
    opcao = input()
        
    while opcao in ['1','2','3','4','5','6','7']:
        if opcao == '1':
            print('')
            for i in perdas_ativas:
                print(i[0])
            print('\nAperte Enter para continuar')
            input()
            menu()
            opcao = input()
            print('')
        if opcao == '2':
            print('')
            for i in perdas_reativas:
                print(i[0])
            print('\nAperte Enter para continuar')
            input()
            menu()
            opcao = input()
            print('')
        if opcao == '3':
            print('')
            for i in tensoes_finais_e_angulos:
                print(i[0])
            print('\nAperte Enter para continuar')
            input()
            menu()
            opcao = input()
            print('')
        if opcao == '4':
            print('')
            for i in potencias_ativas:
                print(i[0])
            print('\nAperte Enter para continuar')
            input()
            menu()
            opcao = input()
            print('')
        if opcao == '5':
            print('')
            for i in potencias_reativas:
                print(i[0])
            print('\nAperte Enter para continuar')
            input()
            menu()
            opcao = input()
            print('')
        if opcao == '6':
            print('')
            for i in fluxo_ativo:
                print(i[0])
            print('\nAperte Enter para continuar')
            input()
            menu()
            opcao = input()
            print('')
        if opcao == '7':
            print('')
            for i in fluxo_reativo:
                print(i[0])
            print('\nAperte Enter para continuar')
            input()
            menu()
            opcao = input()
            print('')
    print('\nReiniciar? s/n\n')
    return

reinicio = ''
programa()
while reinicio != 's' or reinicio != 'n':
    reinicio = input()
    if reinicio == 's':
        programa()
    elif reinicio == 'n':
        break
    