#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# # class:subsection

# In[ ]:


class subsection(object):
    def __init__(self, coordinates, manning):
        self.coordinates = coordinates
        self.manning = manning
         
            
    def H2ABSKn(self, H):
        p = self.coordinates
        n = self.manning
        zero = float(0)
        A,B,S = zero, zero, zero
        SN = zero
        
        for i in range(1, len(p)):
                
            dx = p[i] - p[i-1]    
            hb, hf = H - p[i-1][1], H - p[i][1]
            
            if hb <= 0 :
                if hf > 0 :
                    dx_dh = dx[0] / (hf - hb)
                    B += hf * dx_dh
                    A += 0.5 * hf * hf * dx_dh
                    Sp = hf * np.sqrt( dx_dh * dx_dh + 1.0)
                    S +=  Sp
                    SN += Sp * n[i-1]**1.5
            elif hf <= 0 :
                if hb > 0 :
                    dx_dh = dx[0] / (hf - hb)
                    B -= hb * dx_dh
                    A -= 0.5 * hb * hb * dx_dh
                    Sp = hb * np.sqrt(dx_dh * dx_dh + 1.0)
                    S += Sp
                    SN += Sp * n[i-1]**1.5
            else :
                B += dx[0]
                A += 0.5 * dx[0] * (hf + hb)
                Sp = np.sqrt(dx[0]**2 + dx[1]**2)
                S += Sp
                SN += Sp * n[i-1]**1.5
                
        if S <= zero :
            nd = zero
            K = zero
        else:
            nd = (SN/S)**(2.0/3.0)
            K = A**(5.0/3.0)/nd/S**(2.0/3.0)
            
        return A, B, S, K, nd
    
    def velocity(self, Ie, A, B, S, K, nd):
        # manning eq.
        U = np.sqrt(Ie)*(A/S)**(2.0/3.0)/nd
        
        return U


# # class:section

# In[ ]:


class section(object):
    def __init__(self, ps, ns, distance=np.nan):
        self._subsections = [ subsection(ps[i],ns[i]) for i in range(len(ps)) ]
        self.distance = distance
        
        
    def H2ABSKnSub(self, H):
        num = len(self._subsections)
        A = np.zeros(num)
        B = np.zeros(num)
        S = np.zeros(num)
        K = np.zeros(num)
        n = np.zeros(num)
            
        for nump in range(num) : 
            A[nump], B[nump], S[nump], K[nump], n[nump] = self._subsections[nump].H2ABSKn(H)
            
        return A, B, S, K, n
    
    
    def H2ABS(self, H):
        A, B, S, K, n = self.H2ABSKnSub(H)
        return np.sum(A), np.sum(B), np.sum(S)
        
    
    def zbmin(self):
        return min([ss.coordinates[:,1].min() for ss in self._subsections])
    
    
    def A2HBS(self, A, Hini=float(-9999)):
        if Hini< -9990:
            zb = self.zbmin()
            Ht = zb + float(0.01)
        else:
            Ht = Hini
        
        At, Bt, St = self.H2ABS(Ht)
        while np.abs(At - A) > float(10.0**(-8)):
            Ht += (A - At)/Bt
            if zb > Ht : Ht = zb + float(0.0001)
            At, Bt, St = self.H2ABS(Ht)
            
        return Ht, Bt, St
    
    
    def calIeAlphaBetaRcUsubABS(self, Q, H):
        A, B, S, K, n = self.H2ABSKnSub(H)
        
        Ie = Q**2/np.sum(K)**2
        
        num = len(self._subsections)
        Usub = np.zeros(num)
        RA = float(0)
        Alpha = float(0)
        Beta = float(0)
        
        for i in range(num):
            if A[i] > float(0.0) :
                ss = self._subsections[i]
                Usub[i] = ss.velocity(Ie, A[i], B[i], S[i], K[i], n[i])
                RA += A[i]*(A[i]/S[i])**(2/3)
                Alpha += A[i]*Usub[i]**3
                Beta  += A[i]*Usub[i]**2
        
        Asum = np.sum(A)
        Va = Q/Asum
        Alpha = Alpha/Va**3/Asum
        Beta = Beta/Va**2/Asum
        Rc = ( RA/Asum )**1.5
        
        return Ie, Alpha, Beta, Rc, Usub, Asum, np.sum(B), np.sum(S)
    
    
    def calH0ABS(self, Q, ib):
        dh = float(0.5)
        delta = lambda f : np.abs(ib - f)/f
        
        zb = self.zbmin()
        H = zb + float(0.01)
        arr = self.calIeAlphaBetaRcUsubABS(Q, H)
        ie = arr[0]
        
        while delta(ie) > 0.0001:
            if ib < ie:
                H += dh
            else :
                dh *= float(0.5)
                H -= dh
                
            arr = self.calIeAlphaBetaRcUsubABS(Q, H)
            ie = arr[0]
        
        A, B, S = arr[-3], arr[-2], arr[-1]
        return H, A, B, S 
    
    
    def calHcABS(self, Q):
        dh = float(0.5)
        
        def fr(H):
            arr = self.calIeAlphaBetaRcUsubABS(Q, H)
            Alpha = arr[1]
            Rc = arr[3]
            A, B, S = arr[-3], arr[-2], arr[-1]
            return Q/A/np.sqrt( float(9.8)*Rc / Alpha ), A, B, S
        
        dfr = lambda f : np.abs(-float(1.0) + f)/f
        
        zb = self.zbmin()
        H = zb + float(0.01)
        
        frt, A, B, S = fr(H)
        
        while dfr(frt) > 0.0001:
            if frt > float(1.0):
                H += dh
            else :
                dh *= float(0.5)
                H -= dh
    
            frt, A, B, S = fr(H)
            
        return H, A, B, S

