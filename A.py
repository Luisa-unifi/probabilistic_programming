"""
Python proof-of-concept SIMD-parallel implementation of A,
an algorithm to compute guaranteed estimates for the expectation
of random variables defined by probabilistic programs. It is 
based on a small-step semantics for probabilistic programming
described in: 'Guaranteed inference for probabilistic programs:
a small-step operational approach', the guaratees are given in the
form of confidence intervals.


The main function is:  
    
    - def iter_transition(S,e,xl,T,N,eps):
          It returns the exact confidence interval of e with a given confidence eps:
             - S: model describing the probabilistic program we are
                 studying.
             - e: random variable for which we are interested in obtaining 
                 guarantees on the expected value.
             - xl:variables involved in the considered probabilistic model.
             - T:length of paths considered in the computation.
             - N: number of samples for MC estimation.
             - eps: confidence of the interval returned by the function.
            
Examples and experiments are at the end of the script. 

Note: for better efficiency, all experiments are performed using PyPy: 
    https://www.pypy.org/.
"""


import sympy
from sympy import *
import numpy as np
import time
import copy

from numpy import random   
import functools

U = Function('U') 
G = Function('G')  
D = Function('D')  
lst=Function('lst')
One=sympify(1)
tt=sympify(True)
skip=Function('skip')
fail=Function('fail')
nil=Function('nil')
setv =Function('setv') 
drawv=Function('drawv') 
ifte =Function('ifte') 
obs  =Function('obs') 
seq= Function('seq')
rec= Function('rec')
act= Function('act')
whl= Function('whl')


x,y,z=var('x y z')
trunc_norm=lambda*x: random.normal(loc=x[0],scale=x[1])
uniform =  lambda*x: np.random.uniform(low=x[0],high=x[1])

def discreteD(vals,probs):
    n=vals.shape[0]
    p_cum=probs.cumsum(axis=1)
    cum_v=p_cum[:,-1].reshape((n,1))
    p_cum=p_cum/cum_v 
    r=np.random.uniform(size=(n,1))
    mask=(p_cum<=r)
    idx_cols=mask.sum(axis=1)
    return vals[np.arange(n),idx_cols].reshape((n,1))
    
    
import time

############# Translation from CSP-like to CCS-like language ###########
varcount=0
def newvar(name='u'):
    global varcount
    w=sympy.var(name+'_'+str(varcount) ,real=True)    
    varcount+=1
    return w

def CSP2CCS(S,Sc,dc,thresh=0):
    global varcount
    f=S.func
    if f==skip:
        return Sc,dc
    if f==setv:
        return act(S,Sc),dc
    if f==drawv:
        x = S.args[0]
        rho=S.args[1]
        if rho.func==U:
             if len(rho.args)==0:
                 return act(drawv(x,U(0,1)),Sc),dc
             else:
                 return act(S,Sc),dc
        if rho.func==G:
             if len(rho.args)==2:
                 return act(drawv(x,G(*rho.args)),Sc),dc
             else:
                 return act(S,Sc),dc
        elif rho.func==D:
            return act(S,Sc),dc
        else:
            print('Error, distribution not supported')
            return None
    if f==seq:
        args=S.args
        if len(args)==0:
            return Sc, dc
        S1=args[0]
        if len(args)==1:
            return CSP2CCS(S1,Sc,dc,thresh=thresh) 
        SS2=args[1:]
        if S1==skip():
            return CSP2CCS(seq(*SS2),Sc,dc,thresh=thresh)
        else:
            S0,ndc=CSP2CCS(seq(*SS2),Sc,dc,thresh=thresh)
            if len(str(S0))>thresh:
                A=newvar('A')
                ndc[A]=S0
                return CSP2CCS(S1,A,ndc,thresh=thresh)
            else:
                return CSP2CCS(S1,S0,ndc,thresh=thresh)
    if f==ifte:
        phi,S1,S2=S.args
        S11,dc1=CSP2CCS(S1,Sc,dc,thresh=thresh)
        S22,dc2=CSP2CCS(S2,Sc,dc,thresh=thresh)
        return ifte(phi, S11, S22), dc1|dc2
    if f==obs:
        phi  = S.args[0]
        return ifte(phi, Sc, fail()), dc
    if f==whl:
        phi,S1=S.args
        Y=newvar('Y')
        S0,ndc=CSP2CCS(S1,Y,dc,thresh=thresh)
        ndc[Y]=ifte(phi, S0,  Sc )
        return Y, ndc
    print('Syntax error')
    return None

        


def transition_vectorized_CCS(v,p,xl,dc,v_new=None,p_new=None):
    # Create array of masks for each program counter
    pc_set=set(np.concatenate(p)).difference({fail(),nil()})
    masks = np.zeros((len(pc_set), len(v)), dtype=bool)
    for i, k in enumerate(pc_set):
        masks[i] = (p[:,0]== k)  
    zz = np.zeros((len(v),1))
    if type(v_new)!=type(None):
        np.copyto(v_new,v)
        np.copyto(p_new,p)
    else:
        v_new = v.astype(np.float64)
        p_new = p.astype(object)
    # Apply transition function to each mask
    for i, mask in enumerate(masks):
        if not np.any(mask):
            continue  # Skip if no rows correspond to this program counter
        idx=np.where(mask)[0]
        j = idx[0]  # Get index of first row corresponding to this program counter
        S=p[j,0]
        op=S.func
        if op == act:
            alpha,S1=S.args
            if alpha.func==setv:
                xi,e =alpha.args
                i=xl.index(xi)
                fe  =lambdify(xl, Add(e, xl[0], -xl[0], evaluate=False))
                v_new[mask,i] = fe(*v[mask].T)
            if alpha.func == drawv:
                xi,rho =alpha.args
                i=xl.index(xi)
                if rho.func==U:
                    broad_args=tuple(a+z for a in rho.args) # summing a zero vector z to force broadcasting of constants in arguments
                    fargs=lambdify(xl+[z],broad_args)
                    v_new[idx,i]=uniform(*fargs(*np.concatenate([v_new[mask],zz[mask]],axis=1).T))
                elif rho.func==G:
                    broad_args=tuple(a+z for a in rho.args) # summing a zero vector z to force broadcasting of constants in arguments
                    fargs=lambdify(xl+[z],broad_args)
                    v_new[idx,i]=trunc_norm(*fargs(*np.concatenate([v_new[mask],zz[mask]],axis=1).T))
                elif rho.func==D:
                    vals,probs=rho.args # summing a zero vector z to force broadcasting of constants in arguments
                    probs=tuple(a+z for a in probs)
                    val_ar =np.broadcast_to(np.array(list(vals),ndmin=2,dtype=np.float64),(len(idx),len(vals)))
                    ff=lambdify(xl+[z],probs)
                    prob_ar=np.concatenate([ff(*v[idx].T,np.zeros((len(idx),)))],axis=0).T
                    v_new[idx,i:i+1]=discreteD(val_ar,prob_ar)
                else:
                    print('Syntax error')
                    return None
            p_new[mask]=S1           
        elif op == ifte:
            e, S1, S2 = S.args
            fe  =lambdify(xl,e)
            p_new[idx]=np.where(fe(*v_new[idx].T)>0,S1,S2).reshape(p_new[idx].shape)
            v_new[idx],p_new[idx] = transition_vectorized_CCS(v_new[mask],p_new[mask],xl,dc)
        elif S in dc.keys():
            p_new[idx]=dc[S]
            v_new[idx],p_new[idx] = transition_vectorized_CCS(v_new[mask],p_new[mask],xl,dc)
        else:
            print('Error')
            return None
    return v_new,p_new 


def transition_vectorized_CCS_2(v,p,xl,dc,v_new=None,p_new=None):
    mask0 = np.full((p[:,0].shape[0],), True)
    zz = np.zeros((len(v),1))
    if type(v_new)!=type(None):
        np.copyto(v_new,v)
        np.copyto(p_new,p)
    else:
        v_new = v.astype(np.float64)
        p_new = p.astype(object)
    while(True):
        if not np.any(mask0):
            break  
        j=np.where(mask0)[0][0]
        S=p[j,0]
        mask=p[:,0]==S
        idx=np.where(mask)[0]
        if S==nil() or S==fail():
            mask0[idx]=False
            continue
        op=S.func
        if op == act:
            alpha,S1=S.args
            if alpha.func==setv:
                xi,e =alpha.args
                i=xl.index(xi)
                fe  =lambdify(xl, Add(e, xl[0], -xl[0], evaluate=False))
                v_new[mask,i] = fe(*v[mask].T)
            if alpha.func == drawv:
                xi,rho =alpha.args
                i=xl.index(xi)
                if rho.func==U:
                    broad_args=tuple(a+z for a in rho.args) 
                    fargs=lambdify(xl+[z],broad_args)
                    v_new[idx,i]=uniform(*fargs(*np.concatenate([v_new[mask],zz[mask]],axis=1).T))
                elif rho.func==G:
                    broad_args=tuple(a+z for a in rho.args) # summing a zero vector z to force broadcasting of constants in arguments
                    fargs=lambdify(xl+[z],broad_args)
                    v_new[idx,i]=trunc_norm(*fargs(*np.concatenate([v_new[mask],zz[mask]],axis=1).T))
                elif rho.func==D:
                    vals,probs=rho.args # summing a zero vector z to force broadcasting of constants in arguments
                    probs=tuple(a+z for a in probs)
                    val_ar =np.broadcast_to(np.array(list(vals),ndmin=2,dtype=np.float64),(len(idx),len(vals)))
                    ff=lambdify(xl+[z],probs)
                    prob_ar=np.concatenate([ff(*v[idx].T,np.zeros((len(idx),)))],axis=0).T
                    v_new[idx,i:i+1]=discreteD(val_ar,prob_ar)               
                else:
                    print('Syntax error')
                    return None
            p_new[mask]=S1           
        elif op == ifte:
            e, S1, S2 = S.args
            fe  =lambdify(xl,e)
            p_new[idx]=np.where(fe(*v_new[idx].T)>0,S1,S2).reshape(p_new[idx].shape)
            v_new[idx],p_new[idx] = transition_vectorized_CCS_2(v_new[mask],p_new[mask],xl,dc)
        elif S in dc.keys():
            p_new[idx]=dc[S]
            v_new[idx],p_new[idx] = transition_vectorized_CCS_2(v_new[mask],p_new[mask],xl,dc)
        else:
            print(S)
            print('Error')
            return None
        mask0[idx]=False
    return v_new,p_new 

time_pc_set=0
def iter_transition(S,e,xl,T,N,maxe=1,eps=0.05,vers=3,transl=True,dc={},thresh=0):
    if transl:
        S,dc=CSP2CCS(S,nil(),{},thresh)
    start_time=time.time()
    V=np.empty((T,N,len(xl)))
    P=np.empty((T,N,1),dtype=object)
    V[0,:,:]=0
    P[0,:,0]=S
    if vers==1:
        for t in range(T-1):
            _= transition_vectorized_CCS(V[t],P[t],xl,dc,V[t+1],P[t+1])
    elif vers==2:
        for t in range(T-1):
            _= transition_vectorized_CCS_2(V[t],P[t],xl,dc,V[t+1],P[t+1])
    else:
        for t in range(T-1):
            _= transition_vectorized_cached(V[t],P[t],xl,dc,V[t+1],P[t+1])
    final_time=(time.time()-start_time)
    fe=lambdify(xl,e)
    termidx=np.where(P[-1]==nil())[0]
    termW=(P[-1]==nil()).sum()
    liveW=((P[-1]!=fail()) & (P[-1]!=nil())).sum()
    feW=(fe(*V[-1,termidx].T)).sum()
    print("N (n. of samples): ",N)
    print("T (length of paths): ",T)
    print("Expectation of "+str(e)+" (interval estimate): ",[feW/(termW+liveW),(feW+maxe*liveW)/termW] )
    print("Exact confidence interval for given espilon:   ",[(feW-eps)/(termW+liveW+eps),(feW+maxe*liveW+eps)/(termW-eps)] )   
    print("Prob. of termination w/i T (interval estimate): ",[termW/(termW+liveW),(termW+liveW)/termW])
    print("Confidence 1-delta, for "+str(e)+"  and epsilon="+str(eps)+" (Hoeffding bound): ", 1-2*np.exp(-2*N*eps**2/maxe**2))
    print("Confidence 1-delta, for termination and epsilon="+str(eps)+" (Hoeffding bound): ", 1-2*np.exp(-2*N*eps**2))
    print("Rejection rate: ",1-(termW+liveW)/N) 
    print("TOTAL elapsed time   %s seconds -------        " % final_time)      
    return V,P



def transition_vectorized_cached(v, p, xl, dc, v_new=None, p_new=None, cache=None):
    mask0 = np.full((p[:, 0].shape[0],), True)
    zz = np.zeros((len(v), 1))
    if v_new is not None and p_new is not None:
        np.copyto(v_new, v)
        np.copyto(p_new, p)
    else:
        v_new = v.astype(np.float64)
        p_new = p.astype(object)

    if cache is None:
        cache = {}

    # Apply transition function to each mask
    while True:
        if not np.any(mask0):
            break  # Skip if set of program counters is empty
        j = np.where(mask0)[0][0]
        S = p[j, 0]
        mask = p[:, 0] == S
        idx = np.where(mask)[0]
        if S == nil() or S == fail():
            mask0[idx] = False
            continue
        op = S.func
        if op == act:
            alpha, S1 = S.args
            if alpha.func == setv:
                xi, e = alpha.args
                i = xl.index(xi)
                fe = cache.get(e, None)
                if fe is None:
                    fe = lambdify(xl, Add(e, xl[0], -xl[0], evaluate=False))
                    cache[e] = fe
                v_new[mask, i] = fe(*v[mask].T)
            elif alpha.func == drawv:
                xi, rho = alpha.args
                i = xl.index(xi)
                if rho.func == U:
                    broad_args = tuple(a + z for a in rho.args)
                    fargs = cache.get(broad_args, None)
                    if fargs is None:
                        fargs = lambdify(xl + [z], broad_args)
                        cache[broad_args] = fargs
                    v_new[idx, i] = uniform(*fargs(*np.concatenate([v_new[mask], zz[mask]], axis=1).T))
                elif rho.func == G:
                    broad_args = tuple(a + z for a in rho.args)
                    fargs = cache.get(broad_args, None)
                    if fargs is None:
                        fargs = lambdify(xl + [z], broad_args)
                        cache[broad_args] = fargs
                    v_new[idx, i] = trunc_norm(*fargs(*np.concatenate([v_new[mask], zz[mask]], axis=1).T))
                elif rho.func==D:
                    vals,probs=rho.args      
                    val_ar =np.broadcast_to(np.array(list(vals),ndmin=2,dtype=np.float64),(len(idx),len(vals)))
                    probs=tuple(a+z for a in probs) # summing a zero vector z to force broadcasting of constants in arguments   
                    ff = cache.get(probs, None)
                    if ff is None:
                        ff = lambdify(xl + [z], probs)
                        cache[probs] = ff                    
                    prob_ar=np.concatenate([ff(*v[idx].T,np.zeros((len(idx),)))],axis=0).T
                    v_new[idx,i:i+1]=discreteD(val_ar,prob_ar)                             
                else:
                    print('Syntax error')
                    return None
            p_new[mask] = S1
        elif op == ifte:
            e, S1, S2 = S.args
            fe = cache.get(e, None)
            if fe is None:
                fe = lambdify(xl, e)
                cache[e] = fe
            p_new[idx] = np.where(fe(*v_new[idx].T) > 0, S1, S2).reshape(p_new[idx].shape)
            v_new[idx], p_new[idx], cache = transition_vectorized_cached(v_new[mask], p_new[mask], xl, dc, cache=cache)
        elif S in dc.keys():
            p_new[idx] = dc[S]
            v_new[idx], p_new[idx], cache = transition_vectorized_cached(v_new[mask], p_new[mask], xl, dc, cache=cache)
        else:
            print(S)
            print('Error')
            return None
        mask0[idx] = False
    return v_new, p_new, cache



def bern(x,e=1/2,vL=[0,1]):   # Bernoulli with success prob. e (expression)
    S=seq(drawv(x,U()),ifte(x>=e,setv(x,vL[0]),setv(x,vL[1])))
    return S

def bern2(x,e=1/2,vL=[0,1]):   # Same as above, but uses discrete distribution D
    S=drawv(x,D((0,1),(1-e,e)))   
    return S



###############################################################################
###############################-EXPERIMENTS-###################################
###############################################################################




#------------------ Example 1: Pearl's burglar alarm model

#model definition
bavar=var('earthquake, burglary, phoneWorking, maryWakes') 
alarm = (earthquake>0) | (burglary>0)
called = (maryWakes>0)  & (phoneWorking>0)

varcount=0
BA = seq(bern(earthquake,0.001), 
             bern(burglary,0.01) ,   
             ifte(earthquake>0, bern(phoneWorking,0.6), 
                                bern(phoneWorking,0.99)),
             ifte( alarm & (earthquake<1), 
                   bern(maryWakes,0.8), 
                   ifte( (earthquake>0) | (burglary>0), 
                          bern(maryWakes,0.6 ), 
                          bern(maryWakes,0.2)
                          )
                 ),
             obs(called)
            )

# computation of the confidence interval for the expected value of r.v. burglary
V,P=iter_transition(BA,burglary,[earthquake, burglary, phoneWorking, maryWakes],10,1000000,vers=2)

 
#------------------ Example 2: TrueSkill model


#model definition
sk=var('skillA, skillB, skillC perfA1,perfB1,perfB2, perfC2,perfA3,perfC3')
uv=var('u1 u2 u3 u4 u5 u6 u7 u8 u9')
varcount=0
scale=100 
TrueSkill = seq(
                drawv(skillA, G(100/scale,10/scale)),
                drawv(skillB, G(100/scale,10/scale)),
                drawv(skillC, G(100/scale,10/scale)),
                # first game:A vs B, A won
                drawv(perfA1, G(skillA,15/scale)),
                drawv(perfB1, G(skillB,15/scale)),
                obs(perfA1 > perfB1),
                # second game:B vs C, B won
                drawv(perfB2, G(skillB,15/scale)),
                drawv(perfC2, G(skillC,15/scale)),
                obs(perfB2 > perfC2),
                #third game:A vs C, A won
                drawv(perfA3, G(skillA,15/scale)),
                drawv(perfC3, G(skillC,15/scale)),
                obs(perfA3 > perfC3)
                )

# computation of the confidence interval for the expected value of r.v. skillA
V,P=iter_transition(TrueSkill,skillA,[skillA, skillB, skillC,perfA1,perfB1,perfB2, perfC2,perfA3,perfC3],11,500000,vers=2)




#------------------ Example 3: Random walks

var('y u i term')

RW1 = seq(drawv(u,U(0,1)), whl((y < 1)&(y>-1), seq(drawv(y, G(y, 2*u)), setv(i, i + 1))))
# computation of the confidence interval for the expected value of r.v. i>=3
V,P=iter_transition(RW1,i>=3,[y,i,u],200,100000,eps=.005,vers=2)

# computation of the confidence interval for the expected value of r.v. u
RW2 = seq(drawv(u,U(0,1)), whl((y < 1)&(y>-1), seq(drawv(y, G(y, 2*u)), setv(i, i + 1))), obs(i>=3))
V,P=iter_transition(RW2,u,[y,i,u],200,100000,eps=.005,vers=2)





