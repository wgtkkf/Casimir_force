# Casimir force, Soviet Physics 2 73 (1956)
# Force constant calculation by French group method
# Dielectric constant: called from external functions
# integral calculation: Double exponential method & trapezoid method
# Used for a large gap distance, larger than wave length
# Coded by Takuro TOKUNAGA
# Last modified: January 18 2018

import math
import numpy as np
import cmath
import time
import sys
import mpmath
from scipy.integrate import trapz, simps, quad, quadrature, romberg, dblquad
start = time.time()

sys.path.append('../dc/')
from dc_dsi import drude
from dc_ndsi import ndsi
from dc_ndsi_ezzahri import ndsi_ezzahri
from dc_pdsi import pdsi
from dc_si import si
from dc_si_adachi import si_adachi
from dc_sic import sic
from dc_sic_real import sicreal

# Variables
nmax = np.power(10.,2)

# Constants
h = 6.626070040*np.power(10.,-34)/(2*np.pi) # reduced planck constant
c = 299792458 # light speed
kb = 1.38064852*np.power(10.,-23) # boltzmann constant

# Unit conversion: electron volt to joule
ucev = 1.602176620898*np.power(10.,-19)
ucangs = 1.0*np.power(10.,-10) # Angstrom to m
ucnm = 1.0*np.power(10.,-9) # nm to m

# parameters for distance
gapmin = 500*ucnm # 0 nm, [m]
gapmax = 1000*ucnm # [m]
gap = gapmin # distance between two atoms, Initialization
dgap = (gapmax-gapmin)/nmax

# parameters for an atom
lc = 5.43*ucangs # lattice constant, SiC 4.36, Si 5.43
raddi = 111*np.power(10.,-12) # silicon 111 pm
area = np.pi*np.power(raddi,2)

# parameters for integral calculation
temperature = 300 # [K]
ymax = (10*kb*temperature)/h # upper omega

# integral parameters, y loop
sn = 1 # n
sh = np.power(10.,-5) # h, -6
conv = np.power(10.,-6) # convergence criteria, -10
dif = 1
total = 0

# integral parameters, x loop
nxmax = np.power(10.,10)
xmin = 0
xmax = (10*kb*temperature)/h # upper omega correct?
x = xmin
dx = (xmax-xmin)/nxmax

# functions for begin & finish
def begin():
    print ("begin")

def finish():
    print ("finish")

# variables for DE method, 0 to +inf
def phi(sn):
    y = np.exp(0.5*np.pi*np.sinh(sn*sh))
    return y

def dphi(sn):
    y = 0.5*np.pi*np.cosh(sn*sh)*np.exp(0.5*np.pi*np.sinh(sn*sh))
    return y

# integrand of lifshitz force, equation (4.1)
def lifshitz_force(x, y, gap):
    # xi
    xi = (x*c)/(2*(y+1)*gap)

    # s1 & s2
    ipu1 = si_adachi(1j*xi) # dielectric function of media1
    ipu2 = si_adachi(1j*xi) # dielectric function of media2
    temp1 = ipu1-1+np.power(y+1,2)
    temp2 = ipu2-1+np.power(y+1,2)
    s1 = np.sqrt(temp1)
    s2 = np.sqrt(temp2)

    # A, D
    a1 = (s1+y+1)*(s2+y+1)/((s1-y-1)*(s2-y-1)) # A
    a3 = (s1+(y+1)*ipu1)*(s2+(y+1)*ipu2)/((s1-(y+1)*ipu1)*(s2-(y+1)*ipu2)) # D

    # integrand
    term1 = 1/(a1*np.exp(x)-1)
    term2 = 1/(a3*np.exp(x)-1)
    integrand = (np.power(x,3)/np.power(y+1,2))*(term1+term2)

    return integrand

# main
begin()

# file open
f = open('springconstant.txt', 'w')
for i in range(0,int(nmax)+1): # Gap distance loop
    if gap > 0:
        for j in range(0,int(nxmax)+1): # x loop
            # coefficient
            a0 = h/(32*np.power(np.pi,2)*np.power(gap,4))

            # Initialization of new & old
            new = lifshitz_force(x, phi(0), gap)*dphi(0)
            old = 0

            while dif>conv: # y loop
                new = new + lifshitz_force(x, phi(-sn), gap)*dphi(-sn)\
                + lifshitz_force(x, phi(sn), gap)*dphi(sn)

                # conv check
                dif = abs(new-old)

                sn = sn+1
                old = new

                if dif < conv:
                    break

            # fixed x, integral calculation of y
            integral = sh*new

            # reset parameters
            old = 0
            sn = 1
            dif = 1

        # total integral value
        if j==0 or j==nxmax:
            total1 = total1 + integral
        else:
            total1 = total1 + 2*integral

        total2 = 0.5*dx*total1 # trapezoid method

        # force
        force = abs(a0*total)*np.power(lc,2)

        # output of spring constant
        f.write(str(gap/ucnm)) # nm
        f.write(str(' '))
        f.write(str(abs(force))) # force
        f.write('\n')

    # Gap distance update
    gap = gap + dgap

    # reset parameters
    old = 0
    sn = 1
    dif = 1

    # current step number
    #print("step:{:.0f}".format(i))

# file close
f.close()
finish()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
