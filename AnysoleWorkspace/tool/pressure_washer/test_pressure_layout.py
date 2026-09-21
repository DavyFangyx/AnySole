#!/usr/bin/env python3
"""Deterministic 48-cell (4x12) adapter tests; no model code is imported."""
import numpy as np
def grid(x): return np.asarray(x,dtype=np.float32).reshape(4,12)
def pooled(x): return grid(x).reshape(4,4,3).sum(2)
def adapt(left,right): return np.concatenate([pooled(left),pooled(right)],axis=1)
def main():
 z=np.zeros(48); one=np.zeros(48); one[0]=1023; assert pooled(one)[0,0]==1023
 assert np.all(adapt(one,z)[:,4:]==0) and np.all(adapt(z,one)[:,:4]==0)
 c=np.full(48,255); assert np.all(pooled(c)==765)
 toe=np.zeros(48); toe[-12:]=1; heel=np.zeros(48); heel[:12]=1
 assert pooled(toe)[-1].sum()==12 and pooled(heel)[0].sum()==12
 assert np.isclose(adapt(c,c).sum(), 48*255*2)
 for x in (0,1023): assert np.asarray(x/1023,dtype=np.float32)>=0
 print('pressure layout tests: PASS (4x12, left|right, /1023 normalization boundary)'); return 0
if __name__=='__main__': raise SystemExit(main())
