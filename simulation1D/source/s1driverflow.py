#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numba import jit


# # nonuniform flow model
# 
# $$
# \begin{align}
# \frac{d}{dx} \left( \frac{\beta Q^2}{2gA^2} + H \right)= -i_e
# \end{align}
# $$

# In[ ]:


@jit(nopython=True, parallel=False)
def NonUniformflow(sections, Q, Hdb):
    g = float(9.8)
    dhini = float(0.5)
    H = np.empty_like(Q)
    H[0] = Hdb
    arr = sections[0].calIeAlphaBetaRcUsubABS(Q[0], H[0])
    ied = arr[0]
    Ad = arr[-3]
    
    Hd = H[0]
    Qd = Q[0]
    
    for i in range(1, len(Q)):
        d = i - 1
        sc, sd = sections[i], sections[d]
        Qc = Q[i]
        Hc = sc.calHcABS( Qc )[0]
        arr = sc.calIeAlphaBetaRcUsubABS(Qc, Hc)
        iec = arr[0]
        Ac  = arr[-3]
    
        dx = sc.distance - sd.distance
        
        E1 = 0.5/g*Qc**2.0/Ac**2.0 + Hc
        E2 = 0.5/g*Qd**2.0/Ad**2.0 + Hd + 0.5*dx*(ied + iec)
        
        if E2 < E1 :
            H[i] = Hc
        else : 
            Hc = Hc + float(0.001)
            dh = dhini
            for n in range(1000):
                arr = sc.calIeAlphaBetaRcUsubABS(Qc, Hc)
                iec = arr[0]
                Ac  = arr[-3]
    
                E1 = 0.5/g*Qc**2.0/Ac**2.0 + Hc
                E2 = 0.5/g*Qd**2.0/Ad**2.0 + Hd + 0.5*dx*(ied + iec)
                
                if np.abs(E1 - E2) < 0.00001 : 
                    break
                elif E1 > E2 :
                    dh *= float(0.5)
                    Hc -= dh
                else:
                    Hc += dh
                
            H[i] = Hc
            
        Qd, Hd, ied, Ad = Qc, Hc, iec, Ac
                
    return H


# # unsteady flow model
# 
# $$
# \begin{align}
#     &\frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = 0 \\
#     &\frac{\partial Q}{\partial t} + \frac{\partial }{\partial x}\left(\dfrac{\beta Q^2}{A}\right) 
#     + gA \frac{\partial H}{\partial x} + gAi_e = 0
# \end{align}
# $$

# In[ ]:


@jit(nopython=True, parallel=False)
def UnSteadyflow(sections, A, Q, H, Abound, Qbound, dt):
    g = float(9.8)
    imax = len(A)
    Anew, Qnew, Hnew = np.zeros(imax), np.zeros(imax), np.zeros(imax)
    ie = np.zeros(imax)
    Beta = np.zeros(imax)
    
# continuous equation
    for i in range(1, imax-1) : 
        dx = 0.5*(sections[i-1].distance - sections[i+1].distance)
        Anew[i] = A[i] - dt * ( Q[i] - Q[i-1] ) / dx
        
    Anew[imax-1] = Abound
    Anew[0] = Anew[1]
#     Anew[0] = (Anew[1] - A[1]) + A[0]
    
    for i in range(imax) : 
        s = sections[i]
        Hnew[i], _, _ = s.A2HBS(Anew[i], H[i])
        arr = s.calIeAlphaBetaRcUsubABS(Q[i], H[i])
        ie[i] = arr[0]
        Beta[i] = arr[2]
    
# moumentum equation
    for i in range(1, imax-1): 
        ic, im, ip = i, i-1, i+1
        dxp = sections[ic].distance - sections[ip].distance
        dxm = sections[im].distance - sections[ic].distance
        dxc = 0.5*(sections[im].distance - sections[ip].distance)
        
        Cr1 = 0.5*( Q[ic]/A[ic] + Q[ip]/A[ip] )*dt/dxp
        Cr2 = 0.5*( Q[ic]/A[ic] + Q[im]/A[im] )*dt/dxm
        dHdx1 = ( Hnew[ip] - Hnew[ic] ) / dxp
        dHdx2 = ( Hnew[ic] - Hnew[im] ) / dxm
        dHdx = (float(1.0) - Cr1) * dHdx1 + Cr2 * dHdx2
        
        Qnew[ic] = Q[ic] - dt * ( Beta[ic]*Q[ic]**2/A[ic] - Beta[im]*Q[im]**2/A[im] ) / dxc                          - dt * g * Anew[ic] * dHdx                          - dt * g * A[ic] * ie[ic] 
        
    Qnew[imax-1] = Qnew[imax-2]
    Qnew[0] = Qbound
        
    return Anew, Qnew, Hnew

