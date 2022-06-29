import numpy as np
import math as kel4
import matplotlib.pyplot as plt


# No 2B

#Memasukan nilai yang diketahui
k = 1.3806e-23 #konstanta boltzman
m = 3.3471e-27 #massa H2
T = 100 #asumsikan jika suhu 100 K

#Mendefinisikan persamaan dan print hasil
def f1():
    v = ((8*k*T)/(kel4.pi*m))**0.5
    print(f'Maka kecepatan rata-rata partikel H2 yang bergerak pada saat T = {T} K adalah, <v> = {v} m/s')
f1()

def f2():
    vrms = (3*k*T/m)**0.5
    print(f'Maka kecepatan rms partikel H2 yang bergerak pada saat T = {T} K adalah, vrms = {vrms} m/s')
f2()

# No 3B
#Memasukan nilai yang diketahui
y = 10 #meter
x = 0 #meter
θ = 60 #derajat
m = 1.5 #kg
Kd = 1.05
vo = 100 #m/s
g = 9.81 #m/s^2
h = 0.005
t = 0

#Set kondisi inisial
vx = vo*np.cos(np.radians(θ))
vy = vo*np.sin(np.radians(θ))
xf = []
yf = []
vxf = []
vyf = []
tf = []

while (y >= 0) :
    v = np.sqrt((vx**2)+(vy)**2)
    ax = -Kd*v*vx
    ay = -g - Kd*v*vy
    xf.append(x)
    yf.append(y)
    vxf.append(vx)
    vyf.append(vy)
    tf.append(t)
   
    x_half = x + vx*h/2
    y_half = y +vy*h/2
    
    vx_half = vx + ax*h/2
    vy_half = vy + ay*h/2
    
    v_half = np.sqrt((vx_half**2)+(vy_half**2))
    ax_half = -Kd*v_half*vx_half
    ay_half = -g-Kd*v_half*vy_half
    
    x += h*vx_half
    y += h*vy_half
    
    vx += h*ax_half
    vy += h*ay_half
    t += h
    
n = len(tf)
print("t"," "*10,"x", " "*9, "y"," "*8, "vx"," "*8, "vy")
for i in range(n):
    print("%.3f %10.5f  %10.5f  %10.4f   %10.4f" %(tf[i],xf[i],yf[i],vxf[i], vyf[i]))

# No 3C dan 3D
x_1 = 0
y_1 = 10
t_1 = 0
vo_1= 250
h_1 = 0.001
vx_1 = vo_1*np.cos(np.radians(θ))
vy_1 = vo_1*np.sin(np.radians(θ))
xf_1 = []
yf_1 = []
tf_1 = []

while (y_1 >= 0) :
    v_1 = np.sqrt((vx_1**2)+(vy_1)**2)
    ax_1 = -Kd*v_1*vx_1
    ay_1 = -g - Kd*v_1*vy_1
    xf_1.append(x_1)
    yf_1.append(y_1)
    tf_1.append(t_1)
   
    x_half1 = x_1 + vx_1*h_1/2
    y_half1 = y_1 +vy_1*h_1/2
    
    vx_half1 = vx_1 + ax_1*h_1/2
    vy_half1 = vy_1 + ay_1*h_1/2
    
    v_half1 = np.sqrt((vx_half1**2)+(vy_half1**2))
    ax_half1 = -Kd*v_half1*vx_half1
    ay_half1 = -g-Kd*v_half1*vy_half1
    
    x_1 += h_1*vx_half1
    y_1 += h_1*vy_half1
    
    vx_1 += h_1*ax_half1
    vy_1 += h_1*ay_half1
    t_1 += h_1
    
x_2 = 0
y_2 = 10
t_2 = 0
vo_2= 500
h_2 = 0.0001
vx_2 = vo_2*np.cos(np.radians(θ))
vy_2 = vo_2*np.sin(np.radians(θ))
xf_2 = []
yf_2 = []
tf_2 = []

while (y_2 >= 0) :
    v_2 = np.sqrt((vx_2**2)+(vy_2)**2)
    ax_2 = -Kd*v_2*vx_2
    ay_2 = -g - Kd*v_2*vy_2
    xf_2.append(x_2)
    yf_2.append(y_2)
    tf_2.append(t_2)
   
    x_half2 = x_1 + vx_2*h_2/2
    y_half2 = y_1 +vy_2*h_2/2
    
    vx_half2 = vx_2 + ax_2*h_2/2
    vy_half2 = vy_2 + ay_2*h_2/2
    
    v_half2 = np.sqrt((vx_half2**2)+(vy_half2**2))
    ax_half2 = -Kd*v_half2*vx_half2
    ay_half2 = -g-Kd*v_half2*vy_half2
    
    x_2 += h_2*vx_half2
    y_2 += h_2*vy_half2
    
    vx_2 += h_2*ax_half2
    vy_2 += h_2*ay_half2
    t_2 += h_2
    
fig,ax = plt.subplots(1, figsize=(12,8))
plt.plot(xf,yf, label = "v0 = 100 m/s", color = 'red')
plt.plot(xf_1,yf_1, label = "v0 = 250 m/s", color = 'green')
plt.plot(xf_2,yf_2, label = "v0 = 500 m/s", color = 'blue')

plt.legend()
plt.grid()
plt.title("Plot Jalur Gerakan Kotak pada Ketiga Variasi Kecepatan")
plt.xlabel('x(m)')
plt.ylabel('y(m)')
plt.show()

print("Saat kecepatan 100 m/s, jarak pada sumbu x ketika bola jatuh yaitu", xf[-1],"meter", "pada detik ke", tf[-1])
print("Saat kecepatan 250 m/s, jarak pada sumbu x ketika bola jatuh yaitu", xf_1[-1],"meter", "pada detik ke", tf_1[-1])
print("Saat kecepatan 500 m/s, jarak pada sumbu x ketika bola jatuh yaitu", xf_2[-1],"meter", "pada detik ke", tf_2[-1])