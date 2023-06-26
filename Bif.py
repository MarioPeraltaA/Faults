#!/usr/bin/python3



# Bif Fault

import ShortCircuit as SC


import numpy as np

sys = SC.main()
# Positive Seq
#print('\n ** Admittance Positve Seq. ** \n')

sys1 = SC.main01(sys)
#print(sys1.Y)


#print('\n ** Impedance Positve Seq. ** \n')
Z1 = np.linalg.inv(sys1.Y)
#print(Z1)


# Negative Seq
#print('\n ** Admittance Negative Seq. ** \n')

sys2 = SC.main02(sys)
#print(sys2.Y)


#print('\n ** Impedance Negative Seq. ** \n')
Z2 = np.linalg.inv(sys2.Y)
#print(Z2)

# Cero Seq
#print('\n ** Admittance Cero Seq. ** \n')

sys3 = SC.main00(sys)
#print(sys3.Y)


#print('\n ** Impedance Cero Seq. ** \n')
Z0 = np.linalg.inv(sys3.Y)
#print(Z0)

# Dictionary with total diagonal secs

secs_diagonal = {}

impedances = [Z0,Z1,Z2]
names = ['Z0_diagonal','Z1_diagonal', 'Z2_diagonal']

# Iterate over each matrix
for matrix, names in zip(impedances, names):
    values = np.diagonal(matrix)
    secs_diagonal[names] = values

#print('** Diagonal values of Z0, Z1 and Z2')

#print(secs_diagonal)


# Value of ZF
Zf = 0.017087


# Value of Vf
Vf = 1 + 0j


# User

fault = input("Enter the number of the faulty bar or A is you want all bars:")

Secuenc_Currents = {}

Line_Currents_pu = {}

if fault.isdigit():
    fault = int(fault)
    if fault < 1 or fault > len(secs_diagonal['Z0_diagonal']):
        print("Invalid bar number!")
        exit()
    else:
        Z0_fault = secs_diagonal['Z0_diagonal'][fault-1]  
        Z1_fault = secs_diagonal['Z1_diagonal'][fault-1]
        Z2_fault = secs_diagonal['Z2_diagonal'][fault-1]

        print("** Values of Zth of bar ", fault)

        print("Z0 of bar ", fault,":",Z0_fault)
        print("Z1 of bar ", fault,":",Z1_fault)
        print("Z2 of bar ", fault,":",Z2_fault,"\n ")

        # Bif fault

        # Secuence Current
        I0 = 0
        I1 = (Vf)/(Z1_fault + Z2_fault + Zf)
        I2 = -I1

        Secuenc_Currents["Sec Cero"] = I0
        Secuenc_Currents["Sec Positive"] = I1
        Secuenc_Currents["Sec Negative"] = I2

        # Current of each phase pu

        IA = 0
        IB = -1j*np.sqrt(3)*I1
        IC = -IB

        Line_Currents_pu["IA"] = IA
        Line_Currents_pu["IB"] = IB
        Line_Currents_pu["IC"] = IC

        # Current of each phase kA


        print('** Secuence Currents pu***')
        print('I0',  Secuenc_Currents['Sec Cero'],'pu')
        print('I1',  Secuenc_Currents['Sec Positive'],'pu')
        print('I2',  Secuenc_Currents['Sec Negative'],'pu')
        print('------------------------')

        print('** Line Currents pu***')
        print('IApu',  Line_Currents_pu['IA'],'pu')
        print('IBpu',  Line_Currents_pu['IB'],'pu')
        print('ICpu',  Line_Currents_pu['IC'],'pu')
        print('------------------------')


elif fault == 'A':
    fault = {}

    num_bars = len(secs_diagonal['Z0_diagonal'])
    for i in range(num_bars):
        fault[i+1] = {
            'Z0_fault': secs_diagonal['Z0_diagonal'][i],
            'Z1_fault': secs_diagonal['Z1_diagonal'][i],
            'Z2_fault': secs_diagonal['Z2_diagonal'][i],
        }
    for num_bars, values in fault.items():
        print("****Failed bar", num_bars,"****")
        print('Z0', values['Z0_fault'])
        print('Z1', values['Z1_fault'])
        print('Z2', values['Z2_fault'])
        print('------------------------')

        # Bif Fault
        I0 = 0
        I1 = (Vf)/(values['Z1_fault'] + values['Z2_fault'] + Zf)
        I2 = -I1

        Secuenc_Currents[i] = {
            'Sec Cero': I0,
            'Sec Positive': I1,
            'Sec Negative': I2,
        }
        
        print('I0',  Secuenc_Currents[i]['Sec Cero'],'pu')
        print('I1',  Secuenc_Currents[i]['Sec Positive'],'pu')
        print('I2',  Secuenc_Currents[i]['Sec Negative'],'pu')
        print('------------------------')
            
        IA = 0
        IB = -1j*np.sqrt(3)*I1
        IC = -IB

        Line_Currents_pu[i] = {
            'IA': IA,
            'IB': IB,
            'IC': IC,
        }

        print('IApu',  Line_Currents_pu[i]['IA'],'pu')
        print('IBpu',  Line_Currents_pu[i]['IB'],'pu')
        print('ICpu',  Line_Currents_pu[i]['IC'],'pu')
        print('------------------------')
    

else:
    fault != 'A'
    print("Invalid input!")
    exit()