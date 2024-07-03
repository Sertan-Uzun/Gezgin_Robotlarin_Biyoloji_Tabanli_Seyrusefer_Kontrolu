"""controller_2 controller."""

from controller import Robot, Motor
from controller import Camera
from controller import CameraRecognitionObject
from controller import GPS
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import sympy as sp
from sympy import Heaviside, nan
from sympy import exp
import random
from sympy import *
import numpy as np
import math
import pandas as pd
import csv

with open("C:/Users/Sertan/Desktop/results/firing.txt","w") as file1:
    file1.write('')

#Sabitler
sim_time=0.1 #saniye
h=0.001 #2 ms'yi yakalamak içim en az bu rezolüsyonda olmalı
Rm=20*10**1 #Ohm/m^2
V_threshold=-50*10**-3
V_rest=-65*10**-3
E_i=-70*10**-3
E_e=0   
t=sp.symbols('t') #sembolik değişken

tau2=20*10**-3 #saniye
Refrector_time =2  #mili saniye
counter_lim=Refrector_time/(1000*h)

with open("C:/Users/Sertan/Desktop/Results/C2.csv", 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
C2 = np.array(data, dtype=float)

with open("C:/Users/Sertan/Desktop/Results/C3.csv", 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
C3 = np.array(data, dtype=float)

def firing_gen(g,firing): #İletkenlik Fonksiyonunun Üretilmesi
    
    tao1=5*10**-3 #miliSaniye -> saniye
    gsym=30*10**-3 #mili Siemens -> Siemens
    
    t=sp.symbols('t') #sembolik değişken tanımlama
    g+=gsym*Heaviside(t-firing)*exp( -(t-firing) / tao1 )

    return g   
    
def color_rec(r,g,b): #RGB renklerin stringe dönüştürülmesi
    if r==1 and g==0 and b==0:
        color='red'
    elif r==0 and g==1 and b==0:
        color='green'
    elif r==0 and g==0 and b==1:
        color='blue'
    elif r==0 and g==0 and b==0:
        color='black'
    elif r==1 and g==1 and b==1:
        color='white'
    elif r==1 and g==1 and b==0:
        color='yellow'
    elif r==1 and g==0 and b==1:
        color='pink'
    elif r==0.5 and g==0 and b==1:
        color='purple'
    elif r==0.9 and g==0.9 and b==0.9:
        color='wall'
    else:
        color=None
    return color
      
def color_fire(color,t_curr): #Algılanan Renge ait Giriş için sinaptik uyaran oluşturulması
    t=sp.symbols('t')

    red_function=0*t; green_function=0*t; blue_function=0*t; black_function=0*t; 
    white_function=0*t; yellow_function=0*t; pink_function=0*t; purple_function=0*t; 
    
    if color=='red':
        red_function=firing_gen(0,t_curr)
    elif color=='green':
        green_function=firing_gen(0,t_curr)
    elif color=='blue':
        blue_function=firing_gen(0,t_curr)
    elif color=='black':
        black_function=firing_gen(0,t_curr)
    elif color=='white':
        white_function=firing_gen(0,t_curr)
    elif color=='yellow':
        yellow_function=firing_gen(0,t_curr)
    elif color=='pink':
        pink_function=firing_gen(0,t_curr)
    elif color=='purple':
        purple_function=firing_gen(0,t_curr)
    
    return red_function,green_function,blue_function,black_function,white_function,yellow_function,pink_function,purple_function 
        
class LIF:
    def __init__(self,number,v_past,f_time,dead_counter,C,G,V,firing,g_func,g):
        self.number=number #Açı değerinin hesaplanması İçin
        self.v_past= v_past #Bir Önceki Gerilim Değeri
        self.f_time=f_time #Ateşleme Zamanlarını Tutma
        self.dead_counter= dead_counter #Refrektör Zamanını Sayması İçin
        self.C= C #Connectivity Matrisi
        self.G= G #simülasyon süresi boyunca G değerini tutması için
        self.V= V #simülasyon süresi boyunca V değerini tutması için
        self.firing= firing  #ateşlenme bilgisi
        self.g_func=g_func # iletkenlik fonksiyonu 
        self.g=g # o andaki g değeri

#Nöronların Tanımlanması
HDPC1=LIF(1,V_rest,[-1],0,C3[0,:],[0],[V_rest*1000],False,0,0); HDPC2=LIF(2,V_rest,[-1],0,C3[1,:],[0],[V_rest*1000],False,0,0); 
HDPC3=LIF(3,V_rest,[-1],0,C3[2,:],[0],[V_rest*1000],False,0,0); HDPC4=LIF(4,V_rest,[-1],0,C3[3,:],[0],[V_rest*1000],False,0,0); 
HDPC5=LIF(5,V_rest,[-1],0,C3[4,:],[0],[V_rest*1000],False,0,0); HDPC6=LIF(6,V_rest,[-1],0,C3[5,:],[0],[V_rest*1000],False,0,0); 
HDPC7=LIF(7,V_rest,[-1],0,C3[6,:],[0],[V_rest*1000],False,0,0); HDPC8=LIF(8,V_rest,[-1],0,C3[7,:],[0],[V_rest*1000],False,0,0); 
HDPC9=LIF(9,V_rest,[-1],0,C3[8,:],[0],[V_rest*1000],False,0,0); HDPC10=LIF(10,V_rest,[-1],0,C3[9,:],[0],[V_rest*1000],False,0,0); 
HDPC11=LIF(11,V_rest,[-1],0,C3[10,:],[0],[V_rest*1000],False,0,0); HDPC12=LIF(12,V_rest,[-1],0,C3[11,:],[0],[V_rest*1000],False,0,0)
HDPC13=LIF(13,V_rest,[-1],0,C3[12,:],[0],[V_rest*1000],False,0,0); HDPC14=LIF(14,V_rest,[-1],0,C3[13,:],[0],[V_rest*1000],False,0,0); 
HDPC15=LIF(15,V_rest,[-1],0,C3[14,:],[0],[V_rest*1000],False,0,0); HDPC16=LIF(16,V_rest,[-1],0,C3[15,:],[0],[V_rest*1000],False,0,0); 
HDPC17=LIF(17,V_rest,[-1],0,C3[16,:],[0],[V_rest*1000],False,0,0); HDPC18=LIF(18,V_rest,[-1],0,C3[17,:],[0],[V_rest*1000],False,0,0); 

OBJ1=LIF(None,V_rest,[-1],0,C2[0,:],[0],[V_rest*1000],False,0,0); OBJ2=LIF(None,V_rest,[-1],0,C2[1,:],[0],[V_rest*1000],False,0,0); 
OBJ3=LIF(None,V_rest,[-1],0,C2[2,:],[0],[V_rest*1000],False,0,0); OBJ4=LIF(None,V_rest,[-1],0,C2[3,:],[0],[V_rest*1000],False,0,0); 
OBJ5=LIF(None,V_rest,[-1],0,C2[4,:],[0],[V_rest*1000],False,0,0); OBJ6=LIF(None,V_rest,[-1],0,C2[5,:],[0],[V_rest*1000],False,0,0); 
OBJ7=LIF(None,V_rest,[-1],0,C2[6,:],[0],[V_rest*1000],False,0,0); OBJ8=LIF(None,V_rest,[-1],0,C2[7,:],[0],[V_rest*1000],False,0,0); 

PF_I1=LIF(None,V_rest,[],0,None,[0],[V_rest*1000],False,0,0); PF_I2=LIF(None,V_rest,[],0,None,[0],[V_rest*1000],False,0,0); 
PF_I3=LIF(None,V_rest,[],0,None,[0],[V_rest*1000],False,0,0); PF_I4=LIF(None,V_rest,[],0,None,[0],[V_rest*1000],False,0,0); 
PF_I5=LIF(None,V_rest,[],0,None,[0],[V_rest*1000],False,0,0); PF_I6=LIF(None,V_rest,[],0,None,[0],[V_rest*1000],False,0,0); 
PF_I7=LIF(None,V_rest,[],0,None,[0],[V_rest*1000],False,0,0); PF_I8=LIF(None,V_rest,[],0,None,[0],[V_rest*1000],False,0,0); 

PF_E1=LIF(None,V_rest,[-1],0,[0,1,1,1,1,1,1,1],[0],[V_rest*1000],False,0,0); PF_E2=LIF(None,V_rest,[-1],0,[1,0,1,1,1,1,1,1],[0],[V_rest*1000],False,0,0); 
PF_E3=LIF(None,V_rest,[-1],0,[1,1,0,1,1,1,1,1],[0],[V_rest*1000],False,0,0); PF_E4=LIF(None,V_rest,[-1],0,[1,1,1,0,1,1,1,1],[0],[V_rest*1000],False,0,0); 
PF_E5=LIF(None,V_rest,[-1],0,[1,1,1,1,0,1,1,1],[0],[V_rest*1000],False,0,0); PF_E6=LIF(None,V_rest,[-1],0,[1,1,1,1,1,0,1,1],[0],[V_rest*1000],False,0,0); 
PF_E7=LIF(None,V_rest,[-1],0,[1,1,1,1,1,1,0,1],[0],[V_rest*1000],False,0,0); PF_E8=LIF(None,V_rest,[-1],0,[1,1,1,1,1,1,1,0],[0],[V_rest*1000],False,0,0); 

InA=LIF(None,V_rest,[],0,None,[0],[V_rest*1000],False,0,0)
InB=LIF(None,V_rest,[],0,None,[0],[V_rest*1000],False,0,0)
InC=LIF(None,V_rest,[],0,None,[0],[V_rest*1000],False,0,0)
InN=LIF(None,V_rest,[],0,None,[0],[V_rest*1000],False,0,0)

# Nöron Sabitleri
tau_HDPC=20*10**-3; Refr_time_HPDC =2; count_lim_HDPC=Refr_time_HPDC/(1000*h)
tau_OBJ=20*10**-3;  Refr_time_OBJ =2;  count_lim_OBJ=Refr_time_OBJ/(1000*h)
tau_PF=20*10**-3;   Refr_time_PF =2;   count_lim_PF=Refr_time_PF/(1000*h)

tau_Ina=20*10**-3;   Refr_time_Ina =2;   count_lim_Ina=Refr_time_Ina/(1000*h)
tau_Inb=20*10**-3;   Refr_time_Inb=2;    count_lim_Inb=Refr_time_Inb/(1000*h)
tau_Inn=20*10**-3;   Refr_time_Inn =2;   count_lim_Inn=Refr_time_Inn/(1000*h)        

#Inhibitory Nöronların Modellenmesi
def In_Behave(spike, counter, f_time, g_neuron, v_past, t_cur, g_spesific, tau, count_lim): #Inn,Inb ve PF_In Davranışlarını Gerçekleştiren Fonksiyon (son 3 input)
 
    v_val=v_past - (h/tau)*(v_past-V_rest) -g_spesific*(h*Rm/tau)*(v_past-E_e)
     
    if spike==False:
        if v_val>V_threshold:
            spike=True
            v_val=V_threshold
            f_time.append(t_cur*1000) #milisaniye olarak kaydet
            g_neuron=firing_gen(g_neuron,t_cur)
    else:
        v_val=V_rest
        counter+=1
        if counter>=count_lim:
            spike=False
            counter=0

    if g_neuron==0:#İletkenlik Değerlerinin Elde Edilip Kaydedilmesi
        g_val=0
    else:
        g_val=g_neuron.evalf(subs={t:t_cur})
        
    return spike, counter, f_time, g_neuron, v_val ,g_val #g_neuron fonksiyon g_val değer

def Ina_Behave(spike, counter, f_time, g_neuron, v_past, t_cur, g_ORgate, g_Inb, g_Inn): #Ina Modeli
 
    v_val=v_past - (h/tau_Ina)*(v_past-V_rest) -g_ORgate*(h*Rm/tau_Ina)*(v_past-E_e)-g_Inb*(h*Rm/tau_Ina)*(v_past-E_i)-g_Inn*(h*Rm/tau_Ina)*(v_past-E_i)
     
    if spike==False:
        if v_val>V_threshold:
            spike=True
            v_val=V_threshold
            f_time.append(t_cur*1000) #milisaniye olarak kaydet
            g_neuron=firing_gen(g_neuron,t_cur)
    else:
        v_val=V_rest
        counter+=1
        if counter>=count_lim_Ina:
            spike=False
            counter=0

    if g_neuron==0:#İletkenlik Değerlerinin Elde Edilip Kaydedilmesi
        g_val=0
    else:
        g_val=g_neuron.evalf(subs={t:t_cur})
        
    return spike, counter, f_time, g_neuron, v_val, g_val

def Inc_Behave(spike, counter, f_time, g_neuron, v_past, t_cur, g_ORgate): #Ina Modeli
 
    v_val=v_past - (h/tau_Ina)*(v_past-V_rest) -g_ORgate*(h*Rm/tau_Ina)*(v_past-E_e)
     
    if spike==False:
        if v_val>V_threshold:
            spike=True
            v_val=V_threshold
            f_time.append(t_cur*1000) #milisaniye olarak kaydet
            g_neuron=firing_gen(g_neuron,t_cur)
    else:
        v_val=V_rest
        counter+=1
        if counter>=count_lim_Ina:
            spike=False
            counter=0

    if g_neuron==0:#İletkenlik Değerlerinin Elde Edilip Kaydedilmesi
        g_val=0
    else:
        g_val=g_neuron.evalf(subs={t:t_cur})
        
    return spike, counter, f_time, g_neuron, v_val, g_val


#Excitatory Nöronların Modellenmesi
def Exc_Behave(spike, counter, f_time, g_neuron, v_past, t_cur, C, G,  g_inhib, g_inp, tau, count_lim): #OBJ ve HDPC Nöron Modeli (son 6 input)
 
    v_val=v_past - (h/tau)*(v_past-V_rest)  -  (h*Rm/tau)*(np.matmul(C,G))*(v_past-E_e)  -  (h*Rm/tau)*g_inhib*(v_past-E_i)  -  (h*Rm/tau)*g_inp*(v_past-E_e)
     
    if spike==False:
        if v_val>V_threshold:
            spike=True
            v_val=V_threshold
            f_time.append(t_cur*1000) #milisaniye olarak kaydet
            g_neuron=firing_gen(g_neuron,t_cur)
    else:
        v_val=V_rest
        counter+=1
        if counter>=count_lim:
            spike=False
            counter=0

    if g_neuron==0:#İletkenlik Değerlerinin Elde Edilip Kaydedilmesi
        g_val=0
    else:
        g_val=g_neuron.evalf(subs={t:t_cur})
        
    return spike, counter, f_time, g_neuron, v_val, g_val

def HDPC_Behave(spike, counter, f_time, g_neuron, v_past, t_cur, C, G,  g_inhib, g_inC, tau, count_lim): #OBJ ve HDPC Nöron Modeli (son 6 input)
 
    v_val=v_past - (h/tau)*(v_past-V_rest)  -  (h*Rm/tau)*(np.matmul(C,G))*(v_past-E_e)  -  (h*Rm/tau)*g_inhib*(v_past-E_i)  -  (h*Rm/tau)*g_inC*(v_past-E_i)
     
    if spike==False:
        if v_val>V_threshold:
            spike=True
            v_val=V_threshold
            f_time.append(t_cur*1000) #milisaniye olarak kaydet
            g_neuron=firing_gen(g_neuron,t_cur)
    else:
        v_val=V_rest
        counter+=1
        if counter>=count_lim:
            spike=False
            counter=0

    if g_neuron==0:#İletkenlik Değerlerinin Elde Edilip Kaydedilmesi
        g_val=0
    else:
        g_val=g_neuron.evalf(subs={t:t_cur})
        
    return spike, counter, f_time, g_neuron, v_val, g_val

def PFe_Behave(spike, counter, f_time, g_neuron, v_past, t_cur, C, G, g_per, tau, count_lim,g_inp, g_inn): #PF_e Nöron Modeli (son 5 input)
 
    v_val=v_past - (h/tau)*(v_past-V_rest)  -  (h*Rm/tau)*(np.matmul(C,G))*(v_past-E_i) -g_per*(h*Rm/tau)*(v_past-E_e) -  (h*Rm/tau)*g_inp*(v_past-E_e)  -  (h*Rm/tau)*g_inn*(v_past-E_i)
     
    if spike==False:
        if v_val>V_threshold:
            spike=True
            v_val=V_threshold
            f_time.append(t_cur*1000) #milisaniye olarak kaydet
            g_neuron=firing_gen(g_neuron,t_cur)
    else:
        v_val=V_rest
        counter+=1
        if counter>=count_lim:
            spike=False
            counter=0

    if g_neuron==0:#İletkenlik Değerlerinin Elde Edilip Kaydedilmesi
        g_val=0
    else:
        g_val=g_neuron.evalf(subs={t:t_cur})
        
    return spike, counter, f_time, g_neuron, v_val, g_val

#Perceptronların Oluşturulması
class Perceptron:
    def __init__(self,f_time,G,g_cur,firing,counter):
        self.f_time=f_time #Ateşleme Zamanlarını Tutma
        self.G=G
        self.g_cur=g_cur # o andaki g değeri
        self.firing= firing  #ateşlenme bilgisi
        self.counter=counter

Perceptron_OR=Perceptron([],[0],0,False,0)
Perceptron_OR2=Perceptron([],[0],0,False,0)
Perceptron_AND=Perceptron([],[0],0,False,0)

#İstenilen Kapı Davranışlarının Modellenmesi

def OR_Behave(f_time,firing,counter,G,t_curr):

    if  sum(G)>= 25*10**(-3):
            firing=True
            f_time.append(t_curr*1000)
            g_curr=30*10**(-3)
    else:
        g_curr=0
 
    return firing, counter, g_curr

def AND_Behave(f_time,firing,counter,G,t_curr):

    if firing==False:
        if  sum(G)>= 8*23*10**(-3):
            firing=True
            f_time.append(t_curr*1000)

    else:
        counter+=1   
        if counter>=5:
            counter=0
            firing=False

    if counter<=3 and 1<=counter:
        g_curr=30*10**(-3)
    else: 
        g_curr=0
        
    return firing, counter, g_curr   
     
# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = 35
#Motor Instances
left_motor=robot.getDevice('motor_1')
left_motor.setPosition(float('+inf'))
left_motor.setVelocity(0.0)

right_motor=robot.getDevice('motor_2')
right_motor.setPosition(float('+inf'))
right_motor.setVelocity(0.0)

cam_1 = robot.getDevice('cam_1')
cam_1.enable(timestep)
cam_1.recognitionEnable(timestep)

time2=0; time3=0; time3_lim=10; 
t_space=[0]; g_red=[0]; g_green=[0]; g_blue=[0]; g_black=[0]; g_white=[0]; g_yellow=[0]; g_pink=[0]; g_purple=[0];
g_per=0; G_per=[0];G0=[0,0,0,0,0,0,0,0]; #Perception Hatası için gerekli g değerleri
color_n1=None; color_n2=None; color_n3=None;  color_n4=None; 

X_pos=[]; Y_pos=[]; 
gps_1= robot.getDevice('gps_1')
gps_1.enable(timestep)

w_lim=5
sim_time=0
stuck_time_lim=350*30
old_color=None

x_hold=np.zeros((400),float) #x kordinatları
y_hold=np.zeros((400),float) #y kordinatları
pos_check=np.zeros((400),float) #Farkların Bulunması


while robot.step(timestep) != -1:

    #Simülasyon Zamanı
    sim_time+=timestep #ms cinsinde
    minute=math.floor(sim_time/(60*1000))
    second=math.floor(sim_time/1000)-minute*60
    milisecond=sim_time-second*1000 -minute*60*1000
    
    if sim_time>15*60*1000: #15. dakikada durdur
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        break
    
    obj=cam_1.getRecognitionObjects()
    pos=gps_1.getValues()
    X_pos.append(pos[0]); Y_pos.append(pos[1]); 

    #Sıkışma Durumunda Rastgele Dönme
    x_hold=np.roll(x_hold,1, axis=0)
    x_hold[0]=round(float(pos[0]), 4 )
    y_hold=np.roll(y_hold,1, axis=0)
    y_hold[0]=round(float(pos[1]), 4 )

    p=1
    while p<399: #Çember denklemi ile robotun hareket ettiğin algılanması
        if pow((x_hold[p]-x_hold[0]),2 ) + pow((y_hold[p]-y_hold[0]),2 ) <pow(0.08,2):
            pos_check[p]=1
        else:
            pos_check[p]=0
        p+=1
    count= np.count_nonzero(pos_check == 1)

    if count>390 and time2==0 and time3==0 and sim_time>stuck_time_lim: #Takılma olup olamdığının belirlenmesi
        num=random.randint(5,13)
        time2=round(num*20/3)#120 for 360

        msg4 = 'Simulation Time: ' + str(minute) + '.' + str(second) + '.' + str(round(milisecond,3)) + '\n' + ' ' + '\n'
        msg5 ='Robot is stuck.' +  '\n' + 'Random Angle of Rotation: ' + str(num*20) + '\n' 
        print(msg5); print(msg4)
        with open("C:/Users/Sertan/Desktop/results/firing.txt","a") as file1:
            file1.write(msg5); file1.write(msg4)
        stuck_time_lim=sim_time+350*30 #10.5 saniye ihnal et

    if time2!=0:  #Dönme için Yazılan Kod
        left_motor.setVelocity(2)
        right_motor.setVelocity(-2)
        if time2>0:
            time2=time2-1

    elif time3!=0 or len(obj) == 0: #Döndükten sonra robotun kısa bir süre düz ilerlemesi için yazılan kod
        left_motor.setVelocity(2)
        right_motor.setVelocity(2)
        if time3>0:
            time3=time3-1

    else:
        if len(obj) != 0:
            dist= []
        for obje in obj: #En Yakın Objeyi Algıla
            pos=obje.getPosition()
            x=( pos[0]*pos[0]+pos[1]*pos[1])**0.5
            dist.append(x); 
        indx= dist.index(min(dist))
                
        # En yakın objenin özelliklerinin belirlenmesi          
        y=obj[indx].getColors()
        spe_color=color_rec(y[0],y[1],y[2]) #renk
        spe_dist=dist[indx]   #uzaklık
        y=obj[indx].getPositionOnImage(); 
        spe_x=y[0]; #kamera görüntüsüne göre X   [0-60] arası
        spe_y=y[1]  #kamera görüntüsüne göre y   [0-60] arası

        v=1; w=0.5
        if spe_color=='wall':
            cam_pos= 30- spe_x #pozitif ise obje solda, negatif ise sağda
            if cam_pos==0:
                w=w_lim
            else:
                w=-5/cam_pos
        else: 
            cam_pos= 30- spe_x #pozitif ise obje solda, negatif ise sağda
            w=0.1*cam_pos
                
        if w>w_lim:
            w=w_lim
        elif w<-w_lim:
            w=-w_lim    
                
        left_motor.setVelocity(v-w)
        right_motor.setVelocity(v+w)

        if spe_color != 'wall' and spe_dist<0.3: #obje algılanması durumu
            left_motor.setVelocity(0)
            right_motor.setVelocity(0)

            if spe_color!=old_color:
                stuck_time_lim=sim_time+350*30 #10.5 saniye ihnal et
            old_color=spe_color

            msg1= 'Recognized Color: ' + str(spe_color) + '\n'
                  
            #Ağ Çalışmadan Önce Gereken Değişkenleri Sıfırla
            PF_E1.g_func=0; PF_E1.g=0; PF_E1.firing=False; PF_E1.v_past=V_rest; PF_E1.dead_counter=0; 
            PF_E2.g_func=0; PF_E2.g=0; PF_E2.firing=False; PF_E2.v_past=V_rest; PF_E2.dead_counter=0; 
            PF_E3.g_func=0; PF_E3.g=0; PF_E3.firing=False; PF_E3.v_past=V_rest; PF_E3.dead_counter=0; 
            PF_E4.g_func=0; PF_E4.g=0; PF_E4.firing=False; PF_E4.v_past=V_rest; PF_E4.dead_counter=0; 
            PF_E5.g_func=0; PF_E5.g=0; PF_E5.firing=False; PF_E5.v_past=V_rest; PF_E5.dead_counter=0; 
            PF_E6.g_func=0; PF_E6.g=0; PF_E6.firing=False; PF_E6.v_past=V_rest; PF_E6.dead_counter=0; 
            PF_E7.g_func=0; PF_E7.g=0; PF_E7.firing=False; PF_E7.v_past=V_rest; PF_E7.dead_counter=0; 
            PF_E8.g_func=0; PF_E8.g=0; PF_E8.firing=False; PF_E8.v_past=V_rest; PF_E8.dead_counter=0; 
                    
            PF_I1.g_func=0; PF_I1.g=0; PF_I1.firing=False; PF_I1.v_past=V_rest; PF_I1.dead_counter=0; 
            PF_I2.g_func=0; PF_I2.g=0; PF_I2.firing=False; PF_I2.v_past=V_rest; PF_I2.dead_counter=0; 
            PF_I3.g_func=0; PF_I3.g=0; PF_I3.firing=False; PF_I3.v_past=V_rest; PF_I3.dead_counter=0; 
            PF_I4.g_func=0; PF_I4.g=0; PF_I4.firing=False; PF_I4.v_past=V_rest; PF_I4.dead_counter=0; 
            PF_I5.g_func=0; PF_I5.g=0; PF_I5.firing=False; PF_I5.v_past=V_rest; PF_I5.dead_counter=0; 
            PF_I6.g_func=0; PF_I6.g=0; PF_I6.firing=False; PF_I6.v_past=V_rest; PF_I6.dead_counter=0; 
            PF_I7.g_func=0; PF_I7.g=0; PF_I7.firing=False; PF_I7.v_past=V_rest; PF_I7.dead_counter=0; 
            PF_I8.g_func=0; PF_I8.g=0; PF_I8.firing=False; PF_I8.v_past=V_rest; PF_I8.dead_counter=0; 
                     
            OBJ1.g_func=0; OBJ1.g=0; OBJ1.firing=False; OBJ1.v_past=V_rest; OBJ1.dead_counter=0; 
            OBJ2.g_func=0; OBJ2.g=0; OBJ2.firing=False; OBJ2.v_past=V_rest; OBJ2.dead_counter=0; 
            OBJ3.g_func=0; OBJ3.g=0; OBJ3.firing=False; OBJ3.v_past=V_rest; OBJ3.dead_counter=0; 
            OBJ4.g_func=0; OBJ4.g=0; OBJ4.firing=False; OBJ4.v_past=V_rest; OBJ4.dead_counter=0; 
            OBJ5.g_func=0; OBJ5.g=0; OBJ5.firing=False; OBJ5.v_past=V_rest; OBJ5.dead_counter=0; 
            OBJ6.g_func=0; OBJ6.g=0; OBJ6.firing=False; OBJ6.v_past=V_rest; OBJ6.dead_counter=0; 
            OBJ7.g_func=0; OBJ7.g=0; OBJ7.firing=False; OBJ7.v_past=V_rest; OBJ7.dead_counter=0; 
            OBJ8.g_func=0; OBJ8.g=0; OBJ8.firing=False; OBJ8.v_past=V_rest; OBJ8.dead_counter=0; 
                    
            HDPC1.g_func=0; HDPC1.g=0; HDPC1.firing=False; HDPC1.v_past=V_rest; HDPC1.dead_counter=0; 
            HDPC2.g_func=0; HDPC2.g=0; HDPC2.firing=False; HDPC2.v_past=V_rest; HDPC2.dead_counter=0; 
            HDPC3.g_func=0; HDPC3.g=0; HDPC3.firing=False; HDPC3.v_past=V_rest; HDPC3.dead_counter=0; 
            HDPC4.g_func=0; HDPC4.g=0; HDPC4.firing=False; HDPC4.v_past=V_rest; HDPC4.dead_counter=0; 
            HDPC5.g_func=0; HDPC5.g=0; HDPC5.firing=False; HDPC5.v_past=V_rest; HDPC5.dead_counter=0; 
            HDPC6.g_func=0; HDPC6.g=0; HDPC6.firing=False; HDPC6.v_past=V_rest; HDPC6.dead_counter=0; 
            HDPC7.g_func=0; HDPC7.g=0; HDPC7.firing=False; HDPC7.v_past=V_rest; HDPC7.dead_counter=0; 
            HDPC8.g_func=0; HDPC8.g=0; HDPC8.firing=False; HDPC8.v_past=V_rest; HDPC8.dead_counter=0; 
            HDPC9.g_func=0; HDPC9.g=0; HDPC9.firing=False; HDPC9.v_past=V_rest; HDPC9.dead_counter=0; 
            HDPC10.g_func=0; HDPC10.g=0; HDPC10.firing=False; HDPC10.v_past=V_rest; HDPC10.dead_counter=0; 
            HDPC11.g_func=0; HDPC11.g=0; HDPC11.firing=False; HDPC11.v_past=V_rest; HDPC11.dead_counter=0; 
            HDPC12.g_func=0; HDPC12.g=0; HDPC12.firing=False; HDPC12.v_past=V_rest; HDPC12.dead_counter=0; 
            HDPC13.g_func=0; HDPC13.g=0; HDPC13.firing=False; HDPC13.v_past=V_rest; HDPC13.dead_counter=0; 
            HDPC14.g_func=0; HDPC14.g=0; HDPC14.firing=False; HDPC14.v_past=V_rest; HDPC14.dead_counter=0; 
            HDPC15.g_func=0; HDPC15.g=0; HDPC15.firing=False; HDPC15.v_past=V_rest; HDPC15.dead_counter=0; 
            HDPC16.g_func=0; HDPC16.g=0; HDPC16.firing=False; HDPC16.v_past=V_rest; HDPC16.dead_counter=0; 
            HDPC17.g_func=0; HDPC17.g=0; HDPC17.firing=False; HDPC17.v_past=V_rest; HDPC17.dead_counter=0; 
            HDPC18.g_func=0; HDPC18.g=0; HDPC18.firing=False; HDPC18.v_past=V_rest; HDPC18.dead_counter=0; 
                    
            range1=51; i=0; res=0;          
            while i<range1: #Vuru Üreten Sinir Ağının çalışması için Yazılan While Döngüsü

                sim_time+=1

                if i==0:
                    red_function,green_function,blue_function,black_function,white_function,yellow_function,pink_function,purple_function =color_fire(spe_color,t_space[-1]/1000)
                else:
                    t_current=((t_space[-1])/1000) +h; t_space.append((t_space[-1] +1)*h*1000); 
                        
                    red_val=red_function.evalf(subs={t:t_current});        g_red.append(1000*red_val);
                    green_val=green_function.evalf(subs={t:t_current});    g_green.append(1000*green_val);
                    blue_val=blue_function.evalf(subs={t:t_current});      g_blue.append(1000*blue_val);
                    black_val=black_function.evalf(subs={t:t_current});    g_black.append(1000*black_val);
                    white_val=white_function.evalf(subs={t:t_current});    g_white.append(1000*white_val);
                    yellow_val=yellow_function.evalf(subs={t:t_current});  g_yellow.append(1000*yellow_val);
                    pink_val=pink_function.evalf(subs={t:t_current});      g_pink.append(1000*pink_val);
                    purple_val=purple_function.evalf(subs={t:t_current});  g_purple.append(1000*purple_val);
                        
                    G_per.append(0)
                    
                    #1. Katman

                    PF_E1.firing , PF_E1.dead_counter , PF_E1.f_time, PF_E1.g_func, PF_E1.v_past, PF_E1.g = PFe_Behave(PF_E1.firing, PF_E1.dead_counter , PF_E1.f_time, PF_E1.g_func, PF_E1.v_past,t_current,PF_E1.C, G0,g_per,tau_PF, Refr_time_PF,red_val,InN.g)
                    PF_E1.V.append(1000*PF_E1.v_past); PF_E1.G.append(1000*PF_E1.g)

                    PF_I1.firing , PF_I1.dead_counter , PF_I1.f_time, PF_I1.g_func, PF_I1.v_past, PF_I1.g= In_Behave(PF_I1.firing, PF_I1.dead_counter , PF_I1.f_time, PF_I1.g_func,PF_I1.v_past, t_current, PF_E1.g, tau_PF, Refr_time_PF)
                    PF_I1.V.append(1000*PF_I1.v_past); PF_I1.G.append(1000*PF_I1.g)

                    PF_E2.firing , PF_E2.dead_counter , PF_E2.f_time, PF_E2.g_func, PF_E2.v_past, PF_E2.g = PFe_Behave(PF_E2.firing, PF_E2.dead_counter , PF_E2.f_time, PF_E2.g_func, PF_E2.v_past,t_current,PF_E2.C, G0,g_per,tau_PF, Refr_time_PF,green_val,InN.g)
                    PF_E2.V.append(1000*PF_E2.v_past); PF_E2.G.append(1000*PF_E2.g)

                    PF_I2.firing , PF_I2.dead_counter , PF_I2.f_time, PF_I2.g_func, PF_I2.v_past, PF_I2.g= In_Behave(PF_I2.firing, PF_I2.dead_counter , PF_I2.f_time, PF_I2.g_func,PF_I2.v_past, t_current, PF_E2.g, tau_PF, Refr_time_PF)
                    PF_I2.V.append(1000*PF_I2.v_past); PF_I2.G.append(1000*PF_I2.g)

                    PF_E3.firing , PF_E3.dead_counter , PF_E3.f_time, PF_E3.g_func, PF_E3.v_past, PF_E3.g = PFe_Behave(PF_E3.firing, PF_E3.dead_counter , PF_E3.f_time, PF_E3.g_func, PF_E3.v_past,t_current,PF_E3.C, G0,g_per,tau_PF, Refr_time_PF,blue_val,InN.g)
                    PF_E3.V.append(1000*PF_E3.v_past); PF_E3.G.append(1000*PF_E3.g)

                    PF_I3.firing , PF_I3.dead_counter , PF_I3.f_time, PF_I3.g_func, PF_I3.v_past, PF_I3.g= In_Behave(PF_I3.firing, PF_I3.dead_counter , PF_I3.f_time, PF_I3.g_func,PF_I3.v_past, t_current, PF_E3.g, tau_PF, Refr_time_PF)
                    PF_I3.V.append(1000*PF_I3.v_past); PF_I3.G.append(1000*PF_I3.g)

                    PF_E4.firing , PF_E4.dead_counter , PF_E4.f_time, PF_E4.g_func, PF_E4.v_past, PF_E4.g = PFe_Behave(PF_E4.firing, PF_E4.dead_counter , PF_E4.f_time, PF_E4.g_func, PF_E4.v_past,t_current,PF_E4.C, G0,g_per,tau_PF, Refr_time_PF,black_val,InN.g)
                    PF_E4.V.append(1000*PF_E4.v_past); PF_E4.G.append(1000*PF_E4.g)

                    PF_I4.firing , PF_I4.dead_counter , PF_I4.f_time, PF_I4.g_func, PF_I4.v_past, PF_I4.g= In_Behave(PF_I4.firing, PF_I4.dead_counter , PF_I4.f_time, PF_I4.g_func,PF_I4.v_past, t_current, PF_E4.g, tau_PF, Refr_time_PF)
                    PF_I4.V.append(1000*PF_I4.v_past); PF_I4.G.append(1000*PF_I4.g)

                    PF_E5.firing , PF_E5.dead_counter , PF_E5.f_time, PF_E5.g_func, PF_E5.v_past, PF_E5.g = PFe_Behave(PF_E5.firing, PF_E5.dead_counter , PF_E5.f_time, PF_E5.g_func, PF_E5.v_past,t_current,PF_E5.C, G0,g_per,tau_PF, Refr_time_PF,white_val,InN.g)
                    PF_E5.V.append(1000*PF_E5.v_past); PF_E5.G.append(1000*PF_E5.g)

                    PF_I5.firing , PF_I5.dead_counter , PF_I5.f_time, PF_I5.g_func, PF_I5.v_past, PF_I5.g= In_Behave(PF_I5.firing, PF_I5.dead_counter , PF_I5.f_time, PF_I5.g_func,PF_I5.v_past, t_current, PF_E5.g, tau_PF, Refr_time_PF)
                    PF_I5.V.append(1000*PF_I5.v_past); PF_I5.G.append(1000*PF_I5.g)

                    PF_E6.firing , PF_E6.dead_counter , PF_E6.f_time, PF_E6.g_func, PF_E6.v_past, PF_E6.g = PFe_Behave(PF_E6.firing, PF_E6.dead_counter , PF_E6.f_time, PF_E6.g_func, PF_E6.v_past,t_current,PF_E6.C, G0,g_per,tau_PF, Refr_time_PF,yellow_val,InN.g)
                    PF_E6.V.append(1000*PF_E6.v_past); PF_E6.G.append(1000*PF_E6.g)

                    PF_I6.firing , PF_I6.dead_counter , PF_I6.f_time, PF_I6.g_func, PF_I6.v_past, PF_I6.g= In_Behave(PF_I6.firing, PF_I6.dead_counter , PF_I6.f_time, PF_I6.g_func,PF_I6.v_past, t_current, PF_E6.g, tau_PF, Refr_time_PF)
                    PF_I6.V.append(1000*PF_I6.v_past); PF_I6.G.append(1000*PF_I6.g)

                    PF_E7.firing , PF_E7.dead_counter , PF_E7.f_time, PF_E7.g_func, PF_E7.v_past, PF_E7.g = PFe_Behave(PF_E7.firing, PF_E7.dead_counter , PF_E7.f_time, PF_E7.g_func, PF_E7.v_past,t_current,PF_E7.C, G0,g_per,tau_PF, Refr_time_PF,pink_val,InN.g)
                    PF_E7.V.append(1000*PF_E7.v_past); PF_E7.G.append(1000*PF_E7.g)

                    PF_I7.firing , PF_I7.dead_counter , PF_I7.f_time, PF_I7.g_func, PF_I7.v_past, PF_I7.g= In_Behave(PF_I7.firing, PF_I7.dead_counter , PF_I7.f_time, PF_I7.g_func,PF_I7.v_past, t_current, PF_E7.g, tau_PF, Refr_time_PF)
                    PF_I7.V.append(1000*PF_I7.v_past); PF_I7.G.append(1000*PF_I7.g)

                    PF_E8.firing , PF_E8.dead_counter , PF_E8.f_time, PF_E8.g_func, PF_E8.v_past, PF_E8.g = PFe_Behave(PF_E8.firing, PF_E8.dead_counter , PF_E8.f_time, PF_E8.g_func, PF_E8.v_past,t_current,PF_E8.C, G0,g_per,tau_PF, Refr_time_PF,purple_val,InN.g)
                    PF_E8.V.append(1000*PF_E8.v_past); PF_E8.G.append(1000*PF_E8.g)

                    PF_I8.firing , PF_I8.dead_counter , PF_I8.f_time, PF_I8.g_func, PF_I8.v_past, PF_I8.g= In_Behave(PF_I8.firing, PF_I8.dead_counter , PF_I8.f_time, PF_I8.g_func,PF_I8.v_past, t_current, PF_E8.g, tau_PF, Refr_time_PF)
                    PF_I8.V.append(1000*PF_I8.v_past); PF_I8.G.append(1000*PF_I8.g)

                    G0=[PF_I1.g,PF_I2.g,PF_I3.g,PF_I4.g,PF_I5.g,PF_I6.g,PF_I7.g,PF_I8.g]
                    G1=[PF_E1.g,PF_E2.g,PF_E3.g,PF_E4.g,PF_E5.g,PF_E6.g,PF_E7.g,PF_E8.g]    
                        
                    #2. Katman

                    OBJ1.firing, OBJ1.dead_counter, OBJ1.f_time, OBJ1.g_func, OBJ1.v_past, OBJ1.g = Exc_Behave(OBJ1.firing, OBJ1.dead_counter, OBJ1.f_time, OBJ1.g_func, OBJ1.v_past, t_current, OBJ1.C, G1, InA.g, red_val, tau_OBJ, Refr_time_OBJ)
                    OBJ1.V.append(1000*OBJ1.v_past); OBJ1.G.append(1000*OBJ1.g)

                    OBJ2.firing, OBJ2.dead_counter, OBJ2.f_time, OBJ2.g_func, OBJ2.v_past, OBJ2.g = Exc_Behave(OBJ2.firing, OBJ2.dead_counter, OBJ2.f_time, OBJ2.g_func, OBJ2.v_past, t_current, OBJ2.C, G1, InA.g, green_val, tau_OBJ, Refr_time_OBJ)
                    OBJ2.V.append(1000*OBJ2.v_past); OBJ2.G.append(1000*OBJ2.g)

                    OBJ3.firing, OBJ3.dead_counter, OBJ3.f_time, OBJ3.g_func, OBJ3.v_past, OBJ3.g = Exc_Behave(OBJ3.firing, OBJ3.dead_counter, OBJ3.f_time, OBJ3.g_func, OBJ3.v_past, t_current, OBJ3.C, G1, InA.g, blue_val, tau_OBJ, Refr_time_OBJ)
                    OBJ3.V.append(1000*OBJ3.v_past); OBJ3.G.append(1000*OBJ3.g)

                    OBJ4.firing, OBJ4.dead_counter, OBJ4.f_time, OBJ4.g_func, OBJ4.v_past, OBJ4.g = Exc_Behave(OBJ4.firing, OBJ4.dead_counter, OBJ4.f_time, OBJ4.g_func, OBJ4.v_past, t_current, OBJ4.C, G1, InA.g, black_val, tau_OBJ, Refr_time_OBJ)
                    OBJ4.V.append(1000*OBJ4.v_past); OBJ4.G.append(1000*OBJ4.g)

                    OBJ5.firing, OBJ5.dead_counter, OBJ5.f_time, OBJ5.g_func, OBJ5.v_past, OBJ5.g = Exc_Behave(OBJ5.firing, OBJ5.dead_counter, OBJ5.f_time, OBJ5.g_func, OBJ5.v_past, t_current, OBJ5.C, G1, InA.g, white_val, tau_OBJ, Refr_time_OBJ)
                    OBJ5.V.append(1000*OBJ5.v_past); OBJ5.G.append(1000*OBJ5.g)
                    
                    OBJ6.firing, OBJ6.dead_counter, OBJ6.f_time, OBJ6.g_func, OBJ6.v_past, OBJ6.g = Exc_Behave(OBJ6.firing, OBJ6.dead_counter, OBJ6.f_time, OBJ6.g_func, OBJ6.v_past, t_current, OBJ6.C, G1, InA.g, yellow_val, tau_OBJ, Refr_time_OBJ)
                    OBJ6.V.append(1000*OBJ6.v_past); OBJ6.G.append(1000*OBJ6.g)
                    
                    OBJ7.firing, OBJ7.dead_counter, OBJ7.f_time, OBJ7.g_func, OBJ7.v_past, OBJ7.g = Exc_Behave(OBJ7.firing, OBJ7.dead_counter, OBJ7.f_time, OBJ7.g_func, OBJ7.v_past, t_current, OBJ7.C, G1, InA.g, pink_val, tau_OBJ, Refr_time_OBJ)
                    OBJ7.V.append(1000*OBJ7.v_past); OBJ7.G.append(1000*OBJ7.g)
                    
                    OBJ8.firing, OBJ8.dead_counter, OBJ8.f_time, OBJ8.g_func, OBJ8.v_past, OBJ8.g = Exc_Behave(OBJ8.firing, OBJ8.dead_counter, OBJ8.f_time, OBJ8.g_func, OBJ8.v_past, t_current, OBJ8.C, G1, InA.g, purple_val, tau_OBJ, Refr_time_OBJ)
                    OBJ8.V.append(1000*OBJ8.v_past); OBJ8.G.append(1000*OBJ8.g)
                    
                    G2=[OBJ1.g,OBJ2.g,OBJ3.g,OBJ4.g,OBJ5.g,OBJ6.g,OBJ7.g,OBJ8.g]
                        
                    #3. Katman

                    HDPC1.firing, HDPC1.dead_counter, HDPC1.f_time, HDPC1.g_func, HDPC1.v_past, HDPC1.g = HDPC_Behave(HDPC1.firing, HDPC1.dead_counter, HDPC1.f_time, HDPC1.g_func, HDPC1.v_past, t_current, HDPC1.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC1.V.append(1000*HDPC1.v_past); HDPC1.G.append(1000*HDPC1.g)
                    
                    HDPC2.firing, HDPC2.dead_counter, HDPC2.f_time, HDPC2.g_func, HDPC2.v_past, HDPC2.g = HDPC_Behave(HDPC2.firing, HDPC2.dead_counter, HDPC2.f_time, HDPC2.g_func, HDPC2.v_past, t_current, HDPC2.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC2.V.append(1000*HDPC2.v_past); HDPC2.G.append(1000*HDPC2.g)
                        
                    HDPC3.firing, HDPC3.dead_counter, HDPC3.f_time, HDPC3.g_func, HDPC3.v_past, HDPC3.g = HDPC_Behave(HDPC3.firing, HDPC3.dead_counter, HDPC3.f_time, HDPC3.g_func, HDPC3.v_past, t_current, HDPC3.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC3.V.append(1000*HDPC3.v_past); HDPC3.G.append(1000*HDPC3.g)
                    
                    HDPC4.firing, HDPC4.dead_counter, HDPC4.f_time, HDPC4.g_func, HDPC4.v_past, HDPC4.g = HDPC_Behave(HDPC4.firing, HDPC4.dead_counter, HDPC4.f_time, HDPC4.g_func, HDPC4.v_past, t_current, HDPC4.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC4.V.append(1000*HDPC4.v_past); HDPC4.G.append(1000*HDPC4.g)
                    
                    HDPC5.firing, HDPC5.dead_counter, HDPC5.f_time, HDPC5.g_func, HDPC5.v_past, HDPC5.g = HDPC_Behave(HDPC5.firing, HDPC5.dead_counter, HDPC5.f_time, HDPC5.g_func, HDPC5.v_past, t_current, HDPC5.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC5.V.append(1000*HDPC5.v_past); HDPC5.G.append(1000*HDPC5.g)
                    
                    HDPC6.firing, HDPC6.dead_counter, HDPC6.f_time, HDPC6.g_func, HDPC6.v_past, HDPC6.g = HDPC_Behave(HDPC6.firing, HDPC6.dead_counter, HDPC6.f_time, HDPC6.g_func, HDPC6.v_past, t_current, HDPC6.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC6.V.append(1000*HDPC6.v_past); HDPC6.G.append(1000*HDPC6.g)
                    
                    HDPC7.firing, HDPC7.dead_counter, HDPC7.f_time, HDPC7.g_func, HDPC7.v_past, HDPC7.g = HDPC_Behave(HDPC7.firing, HDPC7.dead_counter, HDPC7.f_time, HDPC7.g_func, HDPC7.v_past, t_current, HDPC7.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC7.V.append(1000*HDPC7.v_past); HDPC7.G.append(1000*HDPC7.g)
                    
                    HDPC8.firing, HDPC8.dead_counter, HDPC8.f_time, HDPC8.g_func, HDPC8.v_past, HDPC8.g = HDPC_Behave(HDPC8.firing, HDPC8.dead_counter, HDPC8.f_time, HDPC8.g_func, HDPC8.v_past, t_current, HDPC8.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC8.V.append(1000*HDPC8.v_past); HDPC8.G.append(1000*HDPC8.g)
                    
                    HDPC9.firing, HDPC9.dead_counter, HDPC9.f_time, HDPC9.g_func, HDPC9.v_past, HDPC9.g = HDPC_Behave(HDPC9.firing, HDPC9.dead_counter, HDPC9.f_time, HDPC9.g_func, HDPC9.v_past, t_current, HDPC9.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC9.V.append(1000*HDPC9.v_past); HDPC9.G.append(1000*HDPC9.g)
                    
                    HDPC10.firing, HDPC10.dead_counter, HDPC10.f_time, HDPC10.g_func, HDPC10.v_past, HDPC10.g = HDPC_Behave(HDPC10.firing, HDPC10.dead_counter, HDPC10.f_time, HDPC10.g_func, HDPC10.v_past, t_current, HDPC10.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC10.V.append(1000*HDPC10.v_past); HDPC10.G.append(1000*HDPC10.g)
                    
                    HDPC11.firing, HDPC11.dead_counter, HDPC11.f_time, HDPC11.g_func, HDPC11.v_past, HDPC11.g = HDPC_Behave(HDPC11.firing, HDPC11.dead_counter, HDPC11.f_time, HDPC11.g_func, HDPC11.v_past, t_current, HDPC11.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC11.V.append(1000*HDPC11.v_past); HDPC11.G.append(1000*HDPC11.g)
                    
                    HDPC12.firing, HDPC12.dead_counter, HDPC12.f_time, HDPC12.g_func, HDPC12.v_past, HDPC12.g = HDPC_Behave(HDPC12.firing, HDPC12.dead_counter, HDPC12.f_time, HDPC12.g_func, HDPC12.v_past, t_current, HDPC12.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC12.V.append(1000*HDPC12.v_past); HDPC12.G.append(1000*HDPC12.g)
                    
                    HDPC13.firing, HDPC13.dead_counter, HDPC13.f_time, HDPC13.g_func, HDPC13.v_past, HDPC13.g = HDPC_Behave(HDPC13.firing, HDPC13.dead_counter, HDPC13.f_time, HDPC13.g_func, HDPC13.v_past, t_current, HDPC13.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC13.V.append(1000*HDPC13.v_past); HDPC13.G.append(1000*HDPC13.g)
                    
                    HDPC14.firing, HDPC14.dead_counter, HDPC14.f_time, HDPC14.g_func, HDPC14.v_past, HDPC14.g = HDPC_Behave(HDPC14.firing, HDPC14.dead_counter, HDPC14.f_time, HDPC14.g_func, HDPC14.v_past, t_current, HDPC14.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC14.V.append(1000*HDPC14.v_past); HDPC14.G.append(1000*HDPC14.g)
                    
                    HDPC15.firing, HDPC15.dead_counter, HDPC15.f_time, HDPC15.g_func, HDPC15.v_past, HDPC15.g = HDPC_Behave(HDPC15.firing, HDPC15.dead_counter, HDPC15.f_time, HDPC15.g_func, HDPC15.v_past, t_current, HDPC15.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC15.V.append(1000*HDPC15.v_past); HDPC15.G.append(1000*HDPC15.g)
                    
                    HDPC16.firing, HDPC16.dead_counter, HDPC16.f_time, HDPC16.g_func, HDPC16.v_past, HDPC16.g = HDPC_Behave(HDPC16.firing, HDPC16.dead_counter, HDPC16.f_time, HDPC16.g_func, HDPC16.v_past, t_current, HDPC16.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC16.V.append(1000*HDPC16.v_past); HDPC16.G.append(1000*HDPC16.g)
                    
                    HDPC17.firing, HDPC17.dead_counter, HDPC17.f_time, HDPC17.g_func, HDPC17.v_past, HDPC17.g = HDPC_Behave(HDPC17.firing, HDPC17.dead_counter, HDPC17.f_time, HDPC17.g_func, HDPC17.v_past, t_current, HDPC17.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC17.V.append(1000*HDPC17.v_past); HDPC17.G.append(1000*HDPC17.g)
                    
                    HDPC18.firing, HDPC18.dead_counter, HDPC18.f_time, HDPC18.g_func, HDPC18.v_past, HDPC18.g = HDPC_Behave(HDPC18.firing, HDPC18.dead_counter, HDPC18.f_time, HDPC18.g_func, HDPC18.v_past, t_current, HDPC18.C, G2, InB.g, InC.g, tau_HDPC, Refr_time_HPDC)
                    HDPC18.V.append(1000*HDPC18.v_past); HDPC18.G.append(1000*HDPC18.g)
                    
                    G3=[HDPC1.g,HDPC2.g,HDPC3.g,HDPC4.g,HDPC5.g,HDPC6.g,HDPC7.g,HDPC8.g,HDPC9.g,HDPC10.g,HDPC11.g,HDPC12.g,HDPC13.g,HDPC14.g,HDPC15.g,HDPC16.g,HDPC17.g,HDPC18.g,]
                        
                    #Kapılar
                    Perceptron_OR2.firing, Perceptron_OR2.counter, Perceptron_OR2.g_cur = OR_Behave(Perceptron_OR2.f_time, Perceptron_OR2.firing, Perceptron_OR2.counter, G3, t_current)
                    Perceptron_OR2.G.append(Perceptron_OR2.g_cur*1000)
                    
                    Perceptron_OR.firing, Perceptron_OR.counter, Perceptron_OR.g_cur = OR_Behave(Perceptron_OR.f_time, Perceptron_OR.firing, Perceptron_OR.counter, G2, t_current)
                    Perceptron_OR.G.append(Perceptron_OR.g_cur*1000)
                    
                    Perceptron_AND.firing, Perceptron_AND.counter, Perceptron_AND.g_cur = AND_Behave(Perceptron_AND.f_time, Perceptron_AND.firing, Perceptron_AND.counter, G1, t_current)
                    Perceptron_AND.G.append(Perceptron_AND.g_cur*1000)
                    
                    #Özel Inhibitor Nöronlar
                    InA.firing, InA.dead_counter, InA.f_time, InA.g_func, InA.v_past, InA.g = Ina_Behave(InA.firing, InA.dead_counter, InA.f_time, InA.g_func, InA.v_past, t_current, Perceptron_OR.g_cur, InB.g,InN.g)
                    InA.V.append(1000*InA.v_past); InA.G.append(1000*InA.g)
                    
                    InC.firing, InC.dead_counter, InC.f_time, InC.g_func, InC.v_past, InC.g = Inc_Behave(InC.firing, InC.dead_counter, InC.f_time, InC.g_func, InC.v_past, t_current, Perceptron_OR2.g_cur)
                    InC.V.append(1000*InC.v_past); InC.G.append(1000*InC.g)
                    
                    InB.firing, InB.dead_counter, InB.f_time, InB.g_func, InB.v_past, InB.g = In_Behave(InB.firing, InB.dead_counter, InB.f_time, InB.g_func, InB.v_past, t_current, Perceptron_AND.g_cur,tau_Inb,count_lim_Inb)
                    InB.V.append(1000*InB.v_past); InB.G.append(1000*InB.g)
                    
                    InN.firing, InN.dead_counter, InN.f_time, InN.g_func, InN.v_past, InN.g = In_Behave(InN.firing, InN.dead_counter, InN.f_time, InN.g_func, InN.v_past, t_current, g_per ,tau_Inn,count_lim_Inn)
                    InN.V.append(1000*InN.v_past); InN.G.append(1000*InN.g)
                        
                    #Ağırlıkların Hesaplanması
                    O1_f=[PF_E1.f_time[-1], PF_E2.f_time[-1], PF_E3.f_time[-1], PF_E4.f_time[-1], PF_E5.f_time[-1], PF_E6.f_time[-1], PF_E7.f_time[-1], PF_E8.f_time[-1]]
                    O2_f=[OBJ1.f_time[-1], OBJ2.f_time[-1], OBJ3.f_time[-1], OBJ4.f_time[-1], OBJ5.f_time[-1], OBJ6.f_time[-1], OBJ7.f_time[-1], OBJ8.f_time[-1]]
                    O3_f=[HDPC1.f_time[-1],HDPC2.f_time[-1],HDPC3.f_time[-1],HDPC4.f_time[-1],HDPC5.f_time[-1],HDPC6.f_time[-1],HDPC7.f_time[-1],HDPC8.f_time[-1],HDPC9.f_time[-1],HDPC10.f_time[-1],
                            HDPC11.f_time[-1],HDPC12.f_time[-1],HDPC13.f_time[-1],HDPC14.f_time[-1],HDPC15.f_time[-1],HDPC16.f_time[-1],HDPC17.f_time[-1],HDPC18.f_time[-1]]
                        
                    #Sonucu Al
                    f_check=[HDPC1.firing,HDPC2.firing,HDPC3.firing,HDPC4.firing,HDPC5.firing,HDPC6.firing,HDPC7.firing,HDPC8.firing,HDPC9.firing
                             ,HDPC10.firing,HDPC11.firing,HDPC12.firing,HDPC13.firing,HDPC14.firing,HDPC15.firing,HDPC16.firing,HDPC17.firing,HDPC18.firing]
                    
                    for x in f_check:
                        if x==True and res==0: #Ateşleme Olup Olmadığını Kontrol Et
                            f_neurons = [i + 1 for i, element in enumerate(f_check) if element] #Ateşlenen Nöronlar
                            num=random.choice(f_neurons); i=range1-5; res=1; #Şeçilen Nöron
                            color_n4=color_n3; color_n3=color_n2; color_n2=color_n1; color_n1=spe_color 
                        elif i==range1-1 and res==0:
                            num=random.randint(1,18)
                            msg2 = 'No firing!!!' + '\n' + 'Random HDPC number selected as: ' + str(num) + '\n'
                            msg3 = 'Angle of Rotation: ' + str(num*20) + '\n' 
                            msg4 = 'Simulation Time: ' + str(minute) + '.' + str(second) + '.' + str(round(milisecond,3)) + '\n'+ ' ' + '\n'
                            color_n4=color_n3; color_n3=color_n2; color_n2=color_n1; color_n1=spe_color 

                        if res==1 and i==range1-1:
                            msg2 = 'Random Selected HDPC Number: ' + str(num) + ' in ' + str(f_neurons) + '\n'
                            msg3 = 'Angle of Rotation: ' + str(num*20) + '\n' 
                            msg4 = 'Simulation Time: ' + str(minute) + '.' + str(second) + '.' + str(round(milisecond,3)) + '\n' + ' ' + '\n'   
                        
                i+=1   
            
            print(msg1); print(msg2); print(msg3); print(msg4)
            with open("C:/Users/Sertan/Desktop/results/firing.txt","a") as file1:
                file1.write(msg1); file1.write(msg2); file1.write(msg3); file1.write(msg4)

            #Sonuçları Uygula  
            if color_n1==color_n2 and color_n1==color_n3  and color_n1==color_n4:  #Sıkışma Durumunu
                num=random.randint(1,18)
                time2=round(num*20/3)#120 for 360

                msg4 = 'Simulation Time: ' + str(minute) + '.' + str(second) + '.' + str(round(milisecond,3)) + '\n' + ' ' + '\n'
                msg5 ='Robot is stuck. (same color 4 times in a row)' +  '\n' + 'Random Angle of Rotation: ' + str(num*20) + '\n' 
                print(msg5); print(msg4)
                with open("C:/Users/Sertan/Desktop/results/firing.txt","a") as file1:
                    file1.write(msg5); file1.write(msg4)
            elif spe_color=='green':
                print('Succes')
                with open("C:/Users/Sertan/Desktop/results/firing.txt","a") as file1:
                    file1.write('Succes')

                left_motor.setVelocity(0)
                right_motor.setVelocity(0)
                break   

            #Döndür
            if num==18:
                time2=0; time3=time3_lim; 
            else: 
                time2=round(num*20/3)#120 for 360 
                left_motor.setVelocity(2); right_motor.setVelocity(-2); 
                time3=time3_lim

plt.figure(1); 
plt.subplot(421); plt.plot(t_space ,g_red,'k',label='g_red'); axs=plt.gca(); 
axs.set(ylabel='mG'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(422); plt.plot(t_space ,g_green,'k',label='g_green'); axs=plt.gca(); 
axs.set(ylabel='mG'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(423); plt.plot(t_space ,g_blue,'k',label='g_blue'); axs=plt.gca(); 
axs.set(ylabel='mG'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(424); plt.plot(t_space ,g_black,'k',label='g_black'); axs=plt.gca(); 
axs.set(ylabel='mG'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(425); plt.plot(t_space ,g_white,'k',label='g_white'); axs=plt.gca(); 
axs.set(ylabel='mG'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(426); plt.plot(t_space ,g_yellow,'k',label='g_yellow'); axs=plt.gca(); 
axs.set(ylabel='mG'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(427); plt.plot(t_space ,g_pink,'k',label='g_pink'); axs=plt.gca(); 
axs.set(ylabel='mG'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(428); plt.plot(t_space ,g_purple,'k',label='g_purple'); axs=plt.gca(); 
axs.set(ylabel='mG'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')

figure = plt.gcf(); figure.set_size_inches(16, 10); plt.savefig("C:/Users/Sertan/Desktop/results/input.jpg", dpi = 100)

plt.figure(2); 
plt.subplot(441); plt.plot(t_space ,PF_E1.V,'g',label='PF_E1'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(442);plt.plot(t_space ,PF_E2.V,'g',label='PF_E2');axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(443); plt.plot(t_space ,PF_E3.V,'g',label='PF_E3'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(444);plt.plot(t_space ,PF_E4.V,'g',label='PF_E4');axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(445); plt.plot(t_space ,PF_E5.V,'g',label='PF_E5'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(446); plt.plot(t_space ,PF_E6.V,'g',label='PF_E6'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(447);plt.plot(t_space ,PF_E7.V,'g',label='PF_E7');axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(448); plt.plot(t_space ,PF_E8.V,'g',label='PF_E8'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')

plt.subplot(4,4,9); plt.plot(t_space ,PF_I1.V,'g',label='PF_I1'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,10);plt.plot(t_space ,PF_I2.V,'g',label='PF_I2');axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,11); plt.plot(t_space ,PF_I3.V,'g',label='PF_I3'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,12);plt.plot(t_space ,PF_I4.V,'g',label='PF_I4');axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,13); plt.plot(t_space ,PF_I5.V,'g',label='PF_I5'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,14); plt.plot(t_space ,PF_I6.V,'g',label='PF_I6'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,15);plt.plot(t_space ,PF_I7.V,'g',label='PF_I7');axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,16); plt.plot(t_space ,PF_I8.V,'g',label='PF_I8'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')

figure = plt.gcf(); figure.set_size_inches(16, 10); plt.savefig("C:/Users/Sertan/Desktop/results/layer1_PF_V.jpg", dpi = 100)

plt.figure(3); 
plt.subplot(441); plt.plot(t_space ,PF_E1.G,'m',label='PF_E1'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(442);plt.plot(t_space ,PF_E2.G,'m',label='PF_E2');axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(443); plt.plot(t_space ,PF_E3.G,'m',label='PF_E3'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(444);plt.plot(t_space ,PF_E4.G,'m',label='PF_E4');axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(445); plt.plot(t_space ,PF_E5.G,'m',label='PF_E5'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(446); plt.plot(t_space ,PF_E6.G,'m',label='PF_E6'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(447);plt.plot(t_space ,PF_E7.G,'m',label='PF_E7');axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(448); plt.plot(t_space ,PF_E8.G,'m',label='PF_E8'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')

plt.subplot(4,4,9); plt.plot(t_space ,PF_I1.G,'m',label='PF_I1'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,10);plt.plot(t_space ,PF_I2.G,'m',label='PF_I2');axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,11); plt.plot(t_space ,PF_I3.G,'m',label='PF_I3'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,12);plt.plot(t_space ,PF_I4.G,'m',label='PF_I4');axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,13); plt.plot(t_space ,PF_I5.G,'m',label='PF_I5'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,14); plt.plot(t_space ,PF_I6.G,'m',label='PF_I6'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,15);plt.plot(t_space ,PF_I7.G,'m',label='PF_I7');axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,16); plt.plot(t_space ,PF_I8.G,'m',label='PF_I8'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')

figure = plt.gcf(); figure.set_size_inches(16, 10); plt.savefig("C:/Users/Sertan/Desktop/results/layer1_PF_G.jpg", dpi = 100)

plt.figure(4); 
plt.subplot(441); plt.plot(t_space ,OBJ1.V,'g',label='OBJ1'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(442);plt.plot(t_space ,OBJ2.V,'g',label='OBJ2');axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(443); plt.plot(t_space ,OBJ3.V,'g',label='OBJ3'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(444);plt.plot(t_space ,OBJ4.V,'g',label='OBJ4');axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(445); plt.plot(t_space ,OBJ5.V,'g',label='OBJ5'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(446); plt.plot(t_space ,OBJ6.V,'g',label='OBJ6'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(447);plt.plot(t_space ,OBJ7.V,'g',label='OBJ7');axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(448); plt.plot(t_space ,OBJ8.V,'g',label='OBJ8'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')

plt.subplot(4,4,9); plt.plot(t_space ,OBJ1.G,'m',label='OBJ1'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,10);plt.plot(t_space ,OBJ2.G,'m',label='OBJ2');axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,11); plt.plot(t_space ,OBJ3.G,'m',label='OBJ3'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,12);plt.plot(t_space ,OBJ4.G,'m',label='OBJ4');axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,13); plt.plot(t_space ,OBJ5.G,'m',label='OBJ5'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,14); plt.plot(t_space ,OBJ6.G,'m',label='OBJ6'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,15);plt.plot(t_space ,OBJ7.G,'m',label='OBJ7');axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,4,16); plt.plot(t_space ,OBJ8.G,'m',label='OBJ8'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')

figure = plt.gcf(); figure.set_size_inches(16, 10); plt.savefig("C:/Users/Sertan/Desktop/results/layer2_OBJ_VG.jpg", dpi = 100)

plt.figure(5); 
plt.subplot(451); plt.plot(t_space ,HDPC1.V,'g',label='HDPC1'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(452);plt.plot(t_space ,HDPC2.V,'g',label='HDPC2');axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(453); plt.plot(t_space ,HDPC3.V,'g',label='HDPC3'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(454);plt.plot(t_space ,HDPC4.V,'g',label='HDPC4');axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(455); plt.plot(t_space ,HDPC5.V,'g',label='HDPC5'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(456); plt.plot(t_space ,HDPC6.V,'g',label='HDPC6'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(457);plt.plot(t_space ,HDPC7.V,'g',label='HDPC7');axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(458); plt.plot(t_space ,HDPC8.V,'g',label='HDPC8'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(459); plt.plot(t_space ,HDPC9.V,'g',label='HDPC9'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,10); plt.plot(t_space ,HDPC10.V,'g',label='HDPC10'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,11); plt.plot(t_space ,HDPC11.V,'g',label='HDPC11'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,12); plt.plot(t_space ,HDPC12.V,'g',label='HDPC12'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,13); plt.plot(t_space ,HDPC13.V,'g',label='HDPC13'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,14); plt.plot(t_space ,HDPC14.V,'g',label='HDPC14'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,15); plt.plot(t_space ,HDPC15.V,'g',label='HDPC15'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,16); plt.plot(t_space ,HDPC16.V,'g',label='HDPC16'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,17); plt.plot(t_space ,HDPC17.V,'g',label='HDPC17'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,18); plt.plot(t_space ,HDPC18.V,'g',label='HDPC18'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')

figure = plt.gcf(); figure.set_size_inches(16, 10); plt.savefig("C:/Users/Sertan/Desktop/results/layer3_HDPC_V.jpg", dpi = 100)

plt.figure(6); 
plt.subplot(451); plt.plot(t_space ,HDPC1.G,'m',label='HDPC1'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(452);plt.plot(t_space ,HDPC2.G,'m',label='HDPC2');axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(453); plt.plot(t_space ,HDPC3.G,'m',label='HDPC3'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(454);plt.plot(t_space ,HDPC4.G,'m',label='HDPC4');axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(455); plt.plot(t_space ,HDPC5.G,'m',label='HDPC5'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(456); plt.plot(t_space ,HDPC6.G,'m',label='HDPC6'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(457);plt.plot(t_space ,HDPC7.G,'m',label='HDPC7');axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(458); plt.plot(t_space ,HDPC8.G,'m',label='HDPC8'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(459); plt.plot(t_space ,HDPC9.G,'m',label='HDPC9'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,10); plt.plot(t_space ,HDPC10.G,'m',label='HDPC10'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,11); plt.plot(t_space ,HDPC11.G,'m',label='HDPC11'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,12); plt.plot(t_space ,HDPC12.G,'m',label='HDPC12'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,13); plt.plot(t_space ,HDPC13.G,'m',label='HDPC13'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,14); plt.plot(t_space ,HDPC14.G,'m',label='HDPC14'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,15); plt.plot(t_space ,HDPC15.G,'m',label='HDPC15'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,16); plt.plot(t_space ,HDPC16.G,'m',label='HDPC16'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,17); plt.plot(t_space ,HDPC17.G,'m',label='HDPC17'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(4,5,18); plt.plot(t_space ,HDPC18.G,'m',label='HDPC18'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')

figure = plt.gcf(); figure.set_size_inches(16, 10); plt.savefig("C:/Users/Sertan/Desktop/results/layer3_HDPC_G.jpg", dpi = 100)

plt.figure(7); 
plt.subplot(621); plt.plot(t_space ,Perceptron_OR.G,'b',label='Perceptron_OR'); axs=plt.gca(); 
axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,35); axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(623); plt.plot(t_space ,InA.V,'g',label='InA'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(625); plt.plot(t_space ,InA.G,'m',label='InA'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')

plt.subplot(622); plt.plot(t_space ,Perceptron_OR2.G,'b',label='Perceptron_OR2'); axs=plt.gca(); 
axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,35); axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(624); plt.plot(t_space ,InC.V,'g',label='InC'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(626); plt.plot(t_space ,InC.G,'m',label='InC'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')

plt.subplot(627); plt.plot(t_space ,Perceptron_AND.G,'b',label='Perceptron_AND'); axs=plt.gca(); 
axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,35); axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(6,2,9); plt.plot(t_space ,InB.V,'g',label='InB'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(6,2,11); plt.plot(t_space ,InB.G,'m',label='InB'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')

plt.subplot(628); plt.plot(t_space ,G_per,'k',label='Perception Error'); axs=plt.gca(); 
axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,35); axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(6,2,10); plt.plot(t_space ,InN.V,'g',label='InN'); axs=plt.gca(); 
axs.set(ylabel='mV'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-70,-45)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')
plt.subplot(6,2,12); plt.plot(t_space ,InN.G,'m',label='InN'); axs=plt.gca(); 
axs.set(ylabel='mS'); axs.set_xlim(t_space[0],t_space[len(t_space)-1]); axs.set_ylim(-5,75)
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='upper right')

figure = plt.gcf(); figure.set_size_inches(16, 10); plt.savefig("C:/Users/Sertan/Desktop/results/exstra_layer.jpg", dpi = 100)

plt.figure(8); #Labirentin Manuel Olarak Modellenmesi ve Robotun İzlediği Yolun Çizdirilmesi

for k in range (0, len(X_pos), math.floor(len(X_pos)/10)):
    color1=(0,k/len(X_pos),0)
    plt.plot(X_pos[k:(k+math.floor(len(X_pos)/10)+1)],Y_pos[k:(k+math.floor(len(X_pos)/10)+1)],c=color1)
    if k>1:
        plt.arrow(X_pos[k-1],Y_pos[k-1],(X_pos[k]-X_pos[k-1]),(Y_pos[k]-Y_pos[k-1]), shape='full', lw=0, length_includes_head=True, head_width=.05,color=color1)
    
axs=plt.gca(); 
axs.set(ylabel='y'); axs.set(ylabel='x'); axs.set_xlim(-3,3); axs.set_ylim(-3,3)

plt.plot(X_pos[0],Y_pos[0],'ko'); plt.plot(X_pos[-1],Y_pos[-1],'go')
axs.add_patch(Rectangle((-1.85,-2.55), 0.1 , 1, color = 'black'))
axs.add_patch(Rectangle((-1.85,-1.6), 1.6 , 0.1, color = 'black'))
axs.add_patch(Rectangle((-0.35,-2.55), 0.1, 1, color = 'black'))
axs.add_patch(Rectangle((-0.35,-2.55), 0.9, 0.1, color = 'black'))
axs.add_patch(Rectangle((0.45,-2.55), 0.1, 1, color = 'black'))
axs.add_patch(Rectangle((0.45,-1.65), 1.6, 0.1, color = 'black'))
axs.add_patch(Rectangle((1.95,-1.65), 0.1, 0.8, color = 'black'))
axs.add_patch(Rectangle((0.45,-0.95), 1.6, 0.1, color = 'black'))
axs.add_patch(Rectangle((0.45,-0.95), 0.1, 1.8, color = 'black'))
axs.add_patch(Rectangle((0.45,0.75), 1.1, 0.1, color = 'black'))
axs.add_patch(Rectangle((1.45,0.75), 0.1, 0.9, color = 'black'))
axs.add_patch(Rectangle((0.45,1.55), 1.1, 0.1, color = 'black'))
axs.add_patch(Rectangle((0.45,1.55), 0.1, 0.9, color = 'black'))
axs.add_patch(Rectangle((-0.35,2.35), 0.9, 0.1, color = 'black'))
axs.add_patch(Rectangle((-0.35,1.55), 0.1, 0.9, color = 'black'))
axs.add_patch(Rectangle((-1.35,1.55), 1.1, 0.1, color = 'black'))
axs.add_patch(Rectangle((-1.35,0.75), 0.1, 0.9, color = 'black'))
axs.add_patch(Rectangle((-1.35,0.75), 1.1, 0.1, color = 'black'))
axs.add_patch(Rectangle((-0.35,-0.95), 0.1, 1.8, color = 'black'))
axs.add_patch(Rectangle((-1.85,-0.95), 1.6, 0.1, color = 'black'))
axs.add_patch(Rectangle((-1.85,-0.95), 0.1, 1.3, color = 'black'))
axs.add_patch(Rectangle((-2.55,0.25), 0.8, 0.1, color = 'black'))
axs.add_patch(Rectangle((-2.55,-2.55), 0.1, 2.9, color = 'black'))

axs.add_patch(Rectangle((-2.2,-0.05), 0.1, 0.1, color = 'magenta'))
axs.add_patch(Rectangle((0.05,2.05), 0.1, 0.1, color = 'magenta'))
axs.add_patch(Rectangle((1.1,1.15), 0.1, 0.1, color = 'magenta'))
axs.add_patch(Rectangle((-1.1,1.15), 0.1, 0.1, color = 'magenta'))
axs.add_patch(Rectangle((1.55,-1.3), 0.1, 0.1, color = 'magenta'))
axs.add_patch(Rectangle((0.05,-2.3), 0.1, 0.1, color = 'darkviolet'))
axs.add_patch(Rectangle((-2.2,-2.5), 0.1, 0.1, color = 'green'))
axs.add_patch(Rectangle((0.05,-1.3), 0.1, 0.1, color = 'blue'))
axs.add_patch(Rectangle((-1.65,-1.25), 0.1, 0.1, color = 'blue'))
axs.add_patch(Rectangle((-2.2,-0.75), 0.1, 0.1, color = 'red'))
axs.add_patch(Rectangle((-0.15,0.55), 0.1, 0.1, color = 'red'))
axs.add_patch(Rectangle((0.15,-1.75), 0.1, 0.1, color = 'red'))
axs.add_patch(Rectangle((0.75,-1.1), 0.1, 0.1, color = 'yellow'))
axs.add_patch(Rectangle((0.15,1.15), 0.1, 0.1, color = 'yellow'))
axs.add_patch(Rectangle((-0.85,-1.5), 0.1, 0.1, color = 'black'))
axs.add_patch(Rectangle((0.3,0.05), 0.1, 0.1, color = 'black'))
axs.add_patch(Rectangle((0.6,1.4), 0.1, 0.1, color = 'black'))
axs.add_patch(Rectangle((0.05,-0.4), 0.1, 0.1, color = 'white'))
axs.add_patch(Rectangle((-2.05,-1.65), 0.1, 0.1, color = 'white'))

axs.set_facecolor("tan")

figure = plt.gcf(); figure.set_size_inches(10, 10); plt.savefig("C:/Users/Sertan/Desktop/results/coordinates.jpg", dpi = 100)