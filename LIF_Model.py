import matplotlib.pyplot as plt
import sympy as sp
from sympy import Heaviside, nan
from sympy import exp

f_inh=[0.05,0.4] #Uyarıı Uyaranlar için Ateşleme Zamanları 
f_exc=[0.15, 0.35] #Baskılayıcı Uyaranlar için Ateşleme Zamanları 

#LIF Nöron Sabitleri
tao1=5*10**-3  #saniye
Delay_time =2  #mili saniye
gsym=40*10**-3 #Siemens
Rm=20*10**1    #Ohm/m^2
V_threshold=-50*10**-3;   V_rest=-65*10**-3 #miliVolt->Volt

#EUler Yönteni için Zaman Aralığı
h=0.001 #1ms hesaplama aralığı

#Sinaptik Vuru için Sabitler
tau2=40*10**-3 #saniye
E_e=0;  #Uyarıcı Uyaran içi Sinaptik Potansiyel 
E_i=-70*10**-3  #Baskılayıcı Uyaran içi Sinaptik Potansiyel 

t=sp.symbols('t') #sembolik değişken tanımlama

g_inh=0;i=0 # Baskılayıcı Uyaran için iletkenlik formülünün hesaplanması
while i <= len(f_inh)-1:
    g_inh+=gsym*Heaviside(t-f_inh[i])*exp(-(t-f_inh[i])/tao1)
    i+=1

g_exc=0;i=0 # Uyarıcı Uyaran için iletkenlik formülünün hesaplanması
while i <= len(f_exc)-1:
    g_exc+=gsym*Heaviside(t-f_exc[i])*exp(-(t-f_exc[i])/tao1)
    i+=1

#Başlangıç Koşulları
G_i=[g_inh.evalf(subs={t:0})];    G_e=[g_exc.evalf(subs={t:0})]
t_space=[0];    V=[V_rest] #Başlangıç Koşulu dinlenme durumunda
spike=False # spike oluştuğunu anlamak için
counter=0;   counter_lim=Delay_time/(1000*h)

#Leaky Integrate and Fire Model
#Spike gelmesi durumunda V değeri threshol değerine eşitlenir ve spike değeri True olur
#Daha sonra gecikme zamanı kadar dinlenme geriliminde tutulur. Bunun için counter kullanılır
#Counter=counter_lim durumunda bu işlem gerçekleştirilmiştir ve counter=0 ile başlangıç durumuna dönülür

i=1 #Zaman için Kullanılan Değişken
while i<= (0.25/h +1):
    t_current=h*i; t_space.append(t_current)
    G_i1=g_inh.evalf(subs={t:t_current}); G_i.append(G_i1)
    G_e1=g_exc.evalf(subs={t:t_current}); G_e.append(G_e1)
    if spike==False:
        v=V[i-1]+h*(-(V[i-1]-V_rest)-Rm*(G_e1*(V[i-1]-E_e)+G_i1*(V[i-1]-E_i)))/tau2 #Euler Denklemi
        if v>V_threshold:
            spike=True
            V.append(V_threshold)
        else:
            V.append(v)
    else:
        V.append(V_rest)
        counter+=1
        if counter==counter_lim:
            spike=False
            counter=0
    i+=1

#Grafik için verilerin mili saniye ve mili Siemens değerlerine dönüştürülmesi 
t_space=[x * 1000 for x in t_space] #milisaniye
G_i_graph=[x * 1000 for x in G_i] #mili Siemens
G_e_graph=[x * 1000 for x in G_e] #mili Siemens
V_graph=[x * 1000 for x in V] #Grafik İçin Veri Dönüştürme Volt->miliVolt

plt.figure(1)

plt.subplot(311); plt.plot(t_space ,G_i_graph,'r',label='Gi'); axs=plt.gca()
axs.set_xlim(t_space[0],t_space[len(t_space)-1]) #x_lim
axs.set( ylabel='mili Siemens',title='Baskılayıcı Sinaps için İletkenlik Eğrisi')
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='best')

plt.subplot(312); plt.plot(t_space ,G_e_graph,'b',label='Ge'); axs=plt.gca()
axs.set_xlim(t_space[0],t_space[len(t_space)-1]) #x_lim
axs.set( ylabel='mili Siemens',title='Uyarıcı Sinaps için İletkenlik Eğrisi')
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='best')

plt.subplot(313); plt.plot(t_space ,V_graph,'g',label='V'); axs=plt.gca()
axs.set_xlim(t_space[0],t_space[len(t_space)-1]) #x_lim
axs.set(xlabel='milisaniye', ylabel='mili Volt',title='Nöron Gerilimi')
axs.grid(which="both");axs.minorticks_on();axs.legend(loc='best')

plt.show()  