# -*- coding: utf-8 -*-
"""
Created on Sun May 16 19:19:59 2021

@author: Rebecca Deniz Hussain
"""
# Import necessary files
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import matplotlib.animation as animation

# Welcome statement
print('-'*100 + '\n' + 'Time dependent wavefunction for a particle in an\n' +
'infinite square well with a single Dirac delta potential\n \n' +
'author: Rebecca Deniz Hussain \n' +
'email: rebeccadenizhussain@gmail.com \n' +
'github: https://github.com/rdnzhsn \n' +
'-'*100)

# Parameters
L = 2 # Length of the physical system
g = 15 # Strength of the Dirac delta potential
m = 1 # Mass of the particle, atomic units
h = 1 # Plancks constant hbar, set to 1 for atomic units
N = 2 # Positive integer for the initial wavepacket
n = 2 # Quantum number
q = 1 # Number of wavevectors
dt = 0.005 # Time step
tmax = 10 # Maximum run time

# Defining x axis values for the system
x_total = np.linspace((-L/2),(L/2),1000)


# Normalization conditions for the wavefunctions
A_odd = np.sqrt(2/L) # Normalization for the stationary odd parity wavefunction solutions

def A_even(k): # Normalization condition for the even parity stationary wavefunction solutions
    return np.sqrt(2/(L-(np.sin(k*L)/k)))

# Numerical roots for the even parity wavevector, k
def k_even(g, q): # Solving the numerical roots of the even parity wavevector, k
    if g <= L:
        left = np.pi*q + 1e-12
        right = np.pi*(q + 1) - 1e-12
    else:
        left = np.pi*(q + 1) + 1e-12
        right = np.pi*(q + 2) - 1e-12
    return brentq(lambda r: r/np.tan(r) - g, left, right)

# Calculating the odd parity stationary wavefunction solutions
def odd_parity_wavefunction(n,x):
    return A_odd*np.sin((2*n*x*np.pi)/L)

# Calculating the odd parity energy eigenvalues
def odd_parity_energy_eigenvalue(n):
    return ((h**2)*(np.pi**2)*((2*n)**2))/(2*m*(L**2))

# Calculating the odd parity expansion coefficient
def odd_parity_expansion_coefficient(N): 
    return (-1**N)*(2**(-1/2))

"The quantum number in the odd parity solutions must equal to the positive"
"integer of the initial wavepacket or else the solutions vanish for the"
"time dependent wavefunction."

# Calculating the even parity stationary wavefunction solutions
def even_parity_wavefunction(k,y): # Stationary even parity wavefunction solutions for all of the system
    return A_even(k_even(g,q))*np.sin(k*(np.abs(x_total)-(L/2)))

# Calculating the even parity energy eigenvalues
def even_parity_energy_eigenvalue(k): # The wavevector k, are from the numerical roots solved above
    return (h**2)*(k**2)/(2*m)

# Calculating the even parity expansion coefficient
def even_parity_expansion_coefficient(N,k):
    return (-1/(np.sqrt((L**2)*(1-(np.sin(k*L)/(k*L))))))*(((np.sin(N*np.pi-k*(L/2))/
                ((2*N*np.pi/L)-k))-(np.sin(N*np.pi+k*(L/2))/((2*N*np.pi/L)+k))))

# Solving for the time dependent wavefunction
def O(n,x,N): # Odd parity half, stationary wavefunction multiplied with the expansion coefficient
    return odd_parity_wavefunction(n, x)*odd_parity_expansion_coefficient(N)

def P(n,x,N,t): # Now multiplying with the time phase factor
    return O(n,x,N)*np.exp((-1j*odd_parity_energy_eigenvalue(n)*t)/h)

def E(k,x,N): # Even parity half, stationary wavefunction multiplied with the expansion coefficient
    return even_parity_expansion_coefficient(N, k_even(g,q))*even_parity_wavefunction(k_even(g,q), x)

def F(k,x,N,t): #  Now multiplying with the time phase factor
    return E(k_even(g,q),x,N)*np.exp((-1j*even_parity_energy_eigenvalue(k_even(g,q))*t)/h)

def wavefunction(k,x,N,dt,n): # Final full time dependent wavefunction
    return F(k_even(g,q),x,N,dt)+P(n,x,N,dt)

# Customising the plot
fig = plt.figure(figsize=(10,6))
ax = plt.axes(ylim=(-1.25,1.25))
plt.axvline(x=-1, ymin=-1, ymax=1, color = "grey", linewidth = 0.5,linestyle = "--") # Adding the boundaries at L/2 for the physical system
plt.axvline(x=1, ymin=-1, ymax=1, color = "grey", linewidth = 0.5,linestyle = "--")
plt.axvline(x=0, ymin=-1, ymax=0.95, color = "k", linewidth = 0.3)  # Adding a line to demonstarte where the Dirac delta potential is located
plt.axhline(y=0, xmin=-1, xmax=1, color = "grey", linewidth = 0.5, linestyle = "--")
plt.title('Time dependent wavefunction for a particle in an  \n' +
'infinite square well with a single Dirac delta potential')


line, = ax.plot([],[], color = 'tomato', linewidth = 1)
ax.set_xlabel('$x/L$')
ax.set_ylabel('$\psi(x)$')

def init():
    line.set_data([],[])
    return line,


def animate(i):
    t = dt*i
    line.set_data(x_total,wavefunction(k_even(g,q), x_total, N, t, n))
    return line,

anim = animation.FuncAnimation(fig,animate,init_func=init,frames = 500,interval = 20,blit = True)
plt.show()    

anim.save("ISWSDDP15 Solution.mp4",fps=50,extra_args=['-vcodec','libx264'])










