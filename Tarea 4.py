
# coding: utf-8

# In[11]:

import numpy as np
import matplotlib.pyplot as plt

#Pregunta 2.1 Tarea 4

class Planeta(object):
    '''
    Clase que permite reproducir el movimiento y energia de un planeta en una orbita determinada que precesa, asumiendo un potencial     
    gravitatorio relativista 
    '''
    global G,M,m
    G=1
    M=1
    m=1
    def __init__(self, condicion_inicial, alpha=0):
        '''
        __init__ es un metodo especial que se usa para inicializar las
        instancias de una clase.

        Ej. de uso:
        >> mercurio = Planeta([x0, y0, vx0, vy0])
        >> print(mercurio.alpha)
        >> 0.
        '''
        self.y_actual = condicion_inicial
        self.t_actual = 0
        self.alpha = alpha

    def ecuacion_de_movimiento(self):
        '''
        Implementa la ecuacion de movimiento, como sistema de ecuaciónes de
        primer orden.
        '''
        x, y, vx, vy = self.y_actual #posiciones y velocidades actuales
        # fx = ... = d2x/dt2 = ax = -dU/dx /m
        fx=lambda x,y,t: (2*self.alpha*G*M*x)/((x**2 + y**2)**2) - (G*M*x)/((np.sqrt(x**2 + y**2))**3)
        # fy = ... = d2y/dt2 = ay = -dU/dy /m
        fy=lambda x,y,t: (2*self.alpha*G*M*y)/((x**2 + y**2)**2) - (G*M*y)/((np.sqrt(x**2 + y**2))**3)
        return [vx, vy, fx, fy]

    def avanza_euler(self, dt):
        '''
        Toma la condicion actual del planeta y avanza su posicion y velocidad
        en un intervalo de tiempo dt usando el metodo de Euler explicito. El metodo no retorna nada pero actualiza los valores de las 
        posiciones y velocidades del planeta.
        Recibe un argumento dt que corresponde al paso de tiempo.
        '''
        #metodo de euler explicito
        t0=self.t_actual
        x0,y0,vx0,vy0=self.y_actual
        fx=self.ecuacion_de_movimiento()[2]
        fy=self.ecuacion_de_movimiento()[3]
        vxn=vx0+dt*fx(x0,y0,t0)
        vyn=vy0+dt*fy(x0,y0,t0)
        xn=x0+dt*vxn
        yn=y0+dt*vyn
        self.y_actual=xn,yn,vxn,vyn
        pass

    def avanza_rk4(self, dt):
        '''
        Toma la condicion actual del planeta y avanza su posicion y velocidad
        en un intervalo de tiempo dt usando el metodo de RK4. El metodo no retorna nada pero actualiza los valores de las 
        posiciones y velocidades del planeta.
        Recibe un argumento dt que corresponde al paso de tiempo.
        '''
        t0=self.t_actual
        x0,y0,vx0,vy0=self.y_actual
        fx=self.ecuacion_de_movimiento()[2]
        fy=self.ecuacion_de_movimiento()[3]
        k1x=dt*vx0
        k1y=dt*vy0
        l1x=dt*fx(x0,y0,t0)
        l1y=dt*fy(x0,y0,t0)
        k2x=dt*(vx0+l1x/2.0)
        k2y=dt*(vy0+l1y/2.0)
        l2x=dt*fx(x0+k1x/2.0,y0+k1y/2.0,t0+dt/2.0)
        l2y=dt*fy(x0+k1x/2.0,y0+k1y/2.0,t0+dt/2.0)
        k3x=dt*(vx0+l2x/2.0)
        k3y=dt*(vy0+l2y/2.0)
        l3x=dt*fx(x0+k2x/2.0,y0+k2y/2.0,t0+dt/2.0)
        l3y=dt*fy(x0+k2x/2.0,y0+k2y/2.0,t0+dt/2.0)
        k4x=dt*(vx0+l3x)
        k4y=dt*(vy0+l3y)
        l4x=dt*fx(x0+k3x,y0+k3y,t0+dt)
        l4y=dt*fy(x0+k3x,y0+k3y,t0+dt)
        xn=x0+(k1x+2*k2x+2*k3x+k4x)/6.0
        vxn=vx0+(l1x+2*l2x+2*l3x+l4x)/6.0
        yn=y0+(k1y+2*k2y+2*k3y+k4y)/6.0
        vyn=vy0+(l1y+2*l2y+2*l3y+l4y)/6.0
        self.y_actual=xn,yn,vxn,vyn
        pass

    def avanza_verlet(self, dt):
        '''
        Toma la condicion actual del planeta y avanza su posicion y velocidad
        en un intervalo de tiempo dt usando el metodo de Verlet. El metodo no retorna nada pero actualiza los valores de las 
        posiciones y velocidades del planeta.
        Recibe un argumento dt que corresponde al paso de tiempo.
        '''
        t0=self.t_actual
        x0,y0,vx0,vy0=self.y_actual
        fx=self.ecuacion_de_movimiento()[2]
        fy=self.ecuacion_de_movimiento()[3]
        xn=x0+vx0*dt+(fx(x0,y0,t0)*(dt**2))/2.0
        yn=y0+vy0*dt+(fy(x0,y0,t0)*(dt**2))/2.0
        vxn=vx0+((fx(x0,y0,t0)+fx(xn,yn,t0+dt))*dt)/2.0
        vyn=vy0+((fy(x0,y0,t0)+fy(xn,yn,t0+dt))*dt)/2.0
        self.y_actual=xn,yn,vxn,vyn
        pass

    def energia_total(self):
        '''
        Calcula la energía total del sistema en las condiciones actuales.
        '''
        x0,y0,vx0,vy0=self.y_actual
        E=0.5*m*(vx0**2 + vy0**2) + (self.alpha*G*M*m)/(x0**2 + y0**2) - (G*M*m)/(np.sqrt(x0**2 + y0**2))
        return E
    
#Pregunta 2.2 Tarea 4

condicion_inicial = [10, 0, 0, 0.14] #[x0,y0,vx0,vy0] vy0 approx. vescape
Peuler = Planeta(condicion_inicial,alpha=0)
Prk4= Planeta(condicion_inicial,alpha=0)
Pverlet= Planeta(condicion_inicial,alpha=0)
dt=0.1 #paso
n=10000 #aprox. 5 periodos t=1000 (=> n*dt=60000*0.1=6000)
#arreglo de arreglos con los valores para el metodo de euler,rk4 y verlet respect.
x=np.zeros((n,3)) 
y=np.zeros((n,3))
vx=np.zeros((n,3))
vy=np.zeros((n,3))
E=np.zeros((n,3))
#euler explicito
for i in range(n):
    x[i][0],y[i][0],vx[i][0],vy[i][0]=Peuler.y_actual
    E[i][0]=Peuler.energia_total()
    Peuler.avanza_euler(dt)
#RK4
for j in range(n):
    x[j][1],y[j][1],vx[j][1],vy[j][1]=Prk4.y_actual
    E[j][1]=Prk4.energia_total()
    Prk4.avanza_rk4(dt)
#Verlet
for h in range(n):
    x[h][2],y[h][2],vx[h][2],vy[h][2]=Pverlet.y_actual
    E[h][2]=Pverlet.energia_total()
    Pverlet.avanza_verlet(dt)
t=np.arange(0,n*dt,dt)
fig1=plt.figure(1)
fig1.clf
plt.plot(x[:,0],y[:,0],'r-')
plt.title(r'Orbita por el metodo de Euler Explicito ($\alpha = 0$)')
plt.xlabel(r'Posicion en X')
plt.ylabel(r'Posicion en Y')
plt.grid(True)
fig1.savefig('euler')
fig2=plt.figure(2)
fig2.clf
plt.plot(x[:,1],y[:,1],'b-')
plt.title(r'Orbita por el metodo de Runge-Kutta orden 4 ($\alpha = 0$)')
plt.xlabel(r'Posicion en X')
plt.ylabel(r'Posicion en Y')
plt.grid(True)
fig2.savefig('rk4')
fig3=plt.figure(3)
fig3.clf
plt.plot(x[:,2],y[:,2],'g-')
plt.title(r'Orbita por el metodo de Verlet ($\alpha = 0$)')
plt.xlabel(r'Posicion en X')
plt.ylabel(r'Posicion en Y')
plt.grid(True)
fig3.savefig('verlet')
fig4=plt.figure(4)
fig4.clf
plt.plot(t,E[:,0],'r-')
plt.title(r'Energia de la orbita vs tiempo (metodo Euler Explicito ($\alpha = 0$))')
plt.xlabel(r'Tiempo')
plt.ylabel(r'Energia')
plt.grid(True)
fig4.savefig('energiaeuler')
fig5=plt.figure(5)
fig5.clf
plt.plot(t,E[:,1],'b-')
plt.title(r'Energia de la orbita vs tiempo (metodo Runge-Kutta orden 4 ($\alpha = 0$))')
plt.xlabel(r'Tiempo')
plt.ylabel(r'Energia')
plt.grid(True)
fig5.savefig('energiark4')
fig6=plt.figure(6)
fig6.clf
plt.plot(t,E[:,2],'g-')
plt.title(r'Energia de la orbita vs tiempo (metodo Verlet ($\alpha = 0$))')
plt.xlabel(r'Tiempo')
plt.ylabel(r'Energia')
plt.grid(True)
fig6.savefig('energiaverlet')

#Pregunta 2.3 Tarea 4

condicion_inicial2 = [10, 0, 0, 0.14]
P = Planeta(condicion_inicial2,alpha=10**(-2.808))
dt=0.1 #paso
n=60000 #aprox. 30 periodos t=6000 (=> n*dt=60000*0.1=6000)
#arreglo de arreglos con los valores para el metodo de euler,rk4 y verlet respect.
x808=np.zeros(n) 
y808=np.zeros(n)
vx808=np.zeros(n)
vy808=np.zeros(n)
E808=np.zeros(n)
t808=np.arange(0,n*dt,dt)
afelio=[]
for h in range(n):
    x808[h],y808[h],vx808[h],vy808[h]=P.y_actual
    E808[h]=P.energia_total()
    d=np.sqrt(x808[h]**2+y808[h]**2)
    afelio.append(d) #approx para t=200 o n=2000 se tiene una orbita, entonces para la ultima orbita t=580 o n=5800, se busca la distancia maxima
    P.avanza_verlet(dt)
afeliofinal=max(afelio[5700:6001])#lista donde esta la distancia al afelio final con 200 distancias (se pone desde el 5700 porque el afelio final deberia estar en 5800 o cerca)
indxafeliofinal=afelio.index(afeliofinal)
xafeliofinal=x808[indxafeliofinal]
yafeliofinal=y808[indxafeliofinal]
angprec=np.arctan(yafeliofinal/xafeliofinal)#radianes
tiempoafeliofinal=indxafeliofinal/(10.0)#segundos
velprec=angprec/tiempoafeliofinal#radianes por segundo
fig7=plt.figure(7)
fig7.clf
plt.plot(x808,y808,'g-')
plt.title(r'Orbita por el metodo de Verlet ($\alpha \neq 0$)')
plt.xlabel('Posicion en X')
plt.ylabel('Posicion en Y')
plt.grid(True)
fig7.savefig('verlet808')
fig8=plt.figure(8)
fig8.clf
plt.plot(t808,E808,'g-')
plt.title(r'Energia de la orbita vs tiempo (metodo Verlet ($\alpha \neq 0 $))')
plt.xlabel('Tiempo')
plt.ylabel('Energia')
plt.grid(True)
fig8.savefig('energiaverlet808')
plt.show()
print 'Indice afelio final',indxafeliofinal,'Tiempo afelio final=',tiempoafeliofinal,'Angulo precesion=',angprec,'Velocidad angular precesion=',velprec



# In[ ]:



