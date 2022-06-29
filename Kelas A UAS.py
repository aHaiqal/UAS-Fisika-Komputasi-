# No 2B

import numpy as np
import scipy.integrate as kel7
import matplotlib.pyplot as plt

h = 6.62607004e-27 #konstanta planck
k = 1.38064852e-16 #konstanta boltzman
c = 2.9979e10 #kecepatan cahaya
pi = np.pi

def Planck(T):
    func = lambda x: (x**3)/((np.exp(x))-1)
    integral, error = kel7.quad(func, 0, np.inf)
    return 2*pi*c**2*h*(((k**4*T**4)/(h**4*c**4)))*(integral)

def Boltzmann(T):
    return ((2*pi**5)*(k**4))/((15*c**2)*h**3)

nT = [100, 300, 900]
planck = []
boltzmann = []

for T in nT:
    planck.append(Planck(T))
    boltzmann.append(Boltzmann(T))

print(f'Nilai Planck : {planck}')
print(f'\nNilai Boltzmann : {boltzmann}')

# No 2C
h = 6.62607004e-27 #konstanta planck
k = 1.38064852e-16 #konstanta boltzman
c = 2.9979e10 #kecepatan cahaya
pi = np.pi

def Planck(T):
    func = lambda x: (x**3)/((np.exp(x))-1)
    integral, error = kel7.quad(func, 0, np.inf)
    return 2*pi*c**2*h*(((k**4*T**4)/(h**4*c**4)))*(integral)

def Boltzmann(T):
    return ((2*pi**5)*(k**4))/((15*c**2)*h**3)

nT = range(0, 1200, 10)
planck = []
boltzmann = []

for T in nT:
    planck.append(Planck(T))
    boltzmann.append(Boltzmann(T))

with plt.style.context('Solarize_Light2'):
    plt.plot(nT, planck, 'b', label='Planck')
    plt.plot(nT, boltzmann, 'g', label='Boltzmann')
    plt.grid(True)
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('Nilai Energi')
    plt.show()
    
    
# No 3B
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import display

#konstanta 
G = (6.6743e-11)
M = 1.9891e30
T_me = 88*24*3600
T_venus = 224.7*24*3600
T_bumi = 365.2*24*3600
T_ma = 687*24*3600
titik_x_mat = 0.0
titik_y_mat = 0.0

# Inisial x,y saat t = 0
x0_me = 57.91e9
x0_venus = 108.21e9
x0_bumi = 149.6e9
x0_ma = 227.92e9

y0_me = 0.0
y0_venus = 0.0
y0_bumi = 0.0
y0_ma = 0.0

# Inisial x,y dx/dt, dy.dt saat t = 0
vx0_me = 0.0
vx0_venus = 0.0
vx0_bumi = 0.0
vx0_ma = 0.0

vy0_me = 47.36*(10**3)
vy0_venus = 35.02*(10**3)
vy0_bumi = 29.78*(10**3)
vy0_ma = 24.07*(10**3)

# Set interval 
a = 0.0
b = T_me
N = 10000
h = (b - a)/N


def f(r_me,t):
    x_me = r_me[0]
    vx_me = r_me[1]
    y_me = r_me[2]
    vy_me = r_me[3]
    jarak = np.sqrt(x_me**2 + y_me**2)
    dvx_me = -G*M*x_me/(jarak**3)
    dvy_me = -G*M*y_me/(jarak**3)
    return np.array([vx_me,dvx_me,vy_me,dvy_me])

titik_t_me = np.arange(a,b,h)
#set nilai awal untuk x dan y saat t = 0
titik_x_me = []
titik_y_me = []


# Nilai awal untuk dx/dt dan dy/dt saat t = 0
r_me = np.array([x0_me,vx0_me,y0_me,vy0_me],float)

#RK orde 4
for t in titik_t_me:
    titik_x_me.append(r_me[0])       
    titik_y_me.append(r_me[2])
    k1 = h*f(r_me,t)
    k2 = h*f(r_me+0.5*k1,t+0.5*h)
    k3 = h*f(r_me+0.5*k2,t+0.5*h)
    k4 = h*f(r_me+k3,t+h)
    r_me += (k1+2*k2+2*k3+k4)/6   
xx_me = titik_x_me  
yy_me = titik_y_me 


#orbit venus
#kondisi inisial x,y saat t = 0
x0_venus = 108.21e9
y0_venus = 0.0


# Kondisi awal x,y dx/dt, dy.dt saat t = 0
vx0_venus = 0.0
vy0_venus = 35.0*(10**3)

# Set interval 
a = 0.0
b= T_venus
N = 10000
h = (b - a)/N


def f(r_venus,t):
    x_venus = r_venus[0]
    vx_venus = r_venus[1]
    y_venus = r_venus[2]
    vy_venus = r_venus[3]
    jarak = np.sqrt(x_venus**2 + y_venus**2)
    dvx_venus = -G*M*x_venus/(jarak**3)
    dvy_venus = -G*M*y_venus/(jarak**3)
    return np.array([vx_venus,dvx_venus,vy_venus,dvy_venus])

titik_t_venus = np.arange(a,b,h)
# set nilai awal untuk x dan y saat t = 0
titik_x_venus = []
titik_y_venus = []

#set nilai awal untuk dx/dt dan dy/dt saat t = 0
r_venus = np.array([x0_venus,vx0_venus,y0_venus,vy0_venus],float)

#RK orde 4
for t in titik_t_venus:
    titik_x_venus.append(r_venus[0])        
    titik_y_venus.append(r_venus[2])
    k1 = h*f(r_venus,t)
    k2 = h*f(r_venus+0.5*k1,t+0.5*h)
    k3 = h*f(r_venus+0.5*k2,t+0.5*h)
    k4 = h*f(r_venus+k3,t+h)
    r_venus += (k1+2*k2+2*k3+k4)/6
     
xx_venus = titik_x_venus 
yy_venus = titik_y_venus 

# Orbit Bumi 
#kondisi inisial (x,y) saat t = 0
x0_bumi = 149.6e9
y0_bumi = 0.0

# inisial (x,y) dx/dt, dy.dt saat t = 0
vx0_bumi = 0.0
vy0_bumi = 29.78*(10**3)

#set interval 
a = 0.0
b = T_bumi
N = 1000
h = (b - a)/N


def f(r_bumi,t):
    x_bumi = r_bumi[0]
    vx_bumi = r_bumi[1]
    y_bumi = r_bumi[2]
    vy_bumi = r_bumi[3]
    jarak = np.sqrt(x_bumi**2 + y_bumi**2)
    dvx_bumi = -G*M*x_bumi/(jarak**3)
    dvy_bumi = -G*M*y_bumi/(jarak**3)
    return np.array([vx_bumi,dvx_bumi,vy_bumi,dvy_bumi])

titik_t_bumi = np.arange(a,b,h)
# Nilai awal untuk x dan y saat t = 0
titik_x_bumi = []
titik_y_bumi = []

# Nilai dx/dt dan dy/dt saat t = 0
r_bumi = np.array([x0_bumi,vx0_bumi,y0_bumi,vy0_bumi],float)

#RK orde 4
for t in titik_t_bumi:
    titik_x_bumi.append(r_bumi[0])        
    titik_y_bumi.append(r_bumi[2])
    k1 = h*f(r_bumi,t)
    k2 = h*f(r_bumi+0.5*k1,t+0.5*h)
    k3 = h*f(r_bumi+0.5*k2,t+0.5*h)
    k4 = h*f(r_bumi+k3,t+h)
    r_bumi += (k1+2*k2+2*k3+k4)/6
     
xx_bumi = titik_x_venus 
yy_bumi = titik_y_venus

# Orbit Mars
#kondisi inisial (x,y) saat t = 0
x0_ma = 227.92e9
y0_ma = 0.0

# Kondisi awal (x,y) dx/dt, dy.dt saat t = 0
vx0_ma = 0.0
vy0_ma = 24.07*(10)**3

#set interval 
a = 0.0
b = T_ma
N = 10000
h = (b - a)/N

def f(r_ma,t):
    x_ma = r_ma[0]
    vx_ma = r_ma[1]
    y_ma = r_ma[2]
    vy_ma = r_ma[3]
    jarak = np.sqrt(x_ma**2 + y_ma**2)
    dvx_ma = -G*M*x_ma/(jarak**3)
    dvy_ma = -G*M*y_ma/(jarak**3)
    return np.array([vx_ma,dvx_ma,vy_ma,dvy_ma])

titik_t_ma = np.arange(a,b,h)
#set nilai awal untuk x dan y saat t = 0
titik_x_ma = []
titik_y_ma = []

#set nilai awal untuk dx/dt dan dy/dt saat t = 0
r_ma = np.array([x0_ma,vx0_ma,y0_ma,vy0_ma],float)

#RK orde 4
for t in titik_t_ma:
    titik_x_ma.append(r_ma[0])       
    titik_y_ma.append(r_ma[2])
    k1 = h*f(r_ma,t)
    k2 = h*f(r_ma+0.5*k1,t+0.5*h)
    k3 = h*f(r_ma+0.5*k2,t+0.5*h)
    k4 = h*f(r_ma+k3,t+h)
    r_ma += (k1+2*k2+2*k3+k4)/6
xx_ma = titik_x_ma
yy_ma = titik_y_ma

with plt.style.context('Solarize_Light2'):
    plt.plot(titik_x_mat,titik_y_mat,'yo',label ='matahari' )
    plt.plot(titik_t_me,titik_x_me,'b', label ='merkurius')
    plt.plot(titik_t_venus,titik_x_venus, 'r',label ='venus')
    plt.plot(titik_t_bumi,titik_x_bumi, 'k',label ='bumi')
    plt.plot(titik_t_ma,titik_x_ma, label ='mars')
    plt.grid(True)
    plt.legend()
    plt.title("Orbit Planet terhadap Matahari")
    plt.xlabel('t(detik)')
    plt.show()
    plt.plot(titik_x_mat,titik_y_mat,'yo',label ='matahari' )
    plt.plot(titik_t_me,titik_y_me, 'b',label ='y(t) merkurius')
    plt.plot(titik_t_venus,titik_y_venus, 'r',label ='y(t) venus')
    plt.plot(titik_t_bumi,titik_y_bumi,'k', label ='y(t) bumi')
    plt.plot(titik_t_ma,titik_y_ma, label ='y(t) mars')
    plt.grid(True)
    plt.legend()
    plt.title("Orbit Planet terhadap Matahari")
    plt.xlabel('t(detik)')
    plt.show()
    
    
# No 3C
print('Plot 4 Planet : ')

with plt.style.context('Solarize_Light2'):
    plt.plot(titik_x_mat,titik_y_mat,'yo', label ='matahari' )
    plt.plot(titik_x_me,titik_y_me,'b', label ='merkurius')
    plt.plot(titik_x_venus,titik_y_venus,'r', label ='venus')
    plt.plot(titik_x_bumi,titik_y_bumi,'k', label ='bumi')
    plt.plot(titik_x_ma,titik_y_ma, label ='mars')
    plt.legend()
    plt.grid(True)
    plt.title("Orbit Planet-Planet terhadap Matahari")
    plt.xlabel('t(detik)')
    plt.show()
    
# No 3D
    """Titik Koordinat matahari pada tata surya dengan berdasarkan hasil pemograman di atas 
    bahwa titik koordinat matahari pada sistem tata surya ada pada x = 0 dan y = 0 
    sehingga matahari tepat berada di pusat sistem tata surya
    """