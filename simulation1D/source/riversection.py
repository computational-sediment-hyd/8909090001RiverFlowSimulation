#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import source.riversectionsuper as sect
import numpy as np
# from numba import jitclass   # import the decorator
from numba.experimental import jitclass
from numba import float64, int8  # import the types
from numba import types


# In[ ]:


spec = [
     ('coordinates', float64[:,:])
    ,('manning'  , float64[:])
]

@jitclass(spec)
class subsection(sect.subsection):
    pass


# In[ ]:


spec = [
    ('_subsections',types.List(subsection.class_type.instance_type, reflected=True))
    ,('distance', float64)
]

@jitclass(spec)
class section(sect.section):
    def __init__(self, ps, ns, distance=np.nan):
        self._subsections = [ subsection(ps[i],ns[i]) for i in range(len(ps)) ]
        self.distance = distance

