# -*- coding: utf-8 -*-
"""
Python prototype implementation of TSI, the vectorized MC algorithm
described in 'Guaranteed inference for probabilistic programs:
a parallelisable, small-step operational approach'([1]), Section 6.

In this file we consider a simplified translation T that can be applied 
for statements S of P_0 that rule out either 'while' or 'if-then-else'.

TSI returns guaranteed estimates for the expectation of random 
variables defined by probabilistic programs, guarantees are 
given in the form of confidence intervals.

The implementation is based on TensorFlow and autograph:
https://github.com/tensorflow/tensorflow/blob/master/
tensorflow/python/autograph/g3doc/reference/index.md.



The main functions are:
    
   1. def translate_seq(S,x,ind=''):
          Given a probabilistic program written in language P_0 described in [1]
          it returns its tanslation into TensorFlow.
             - S: model describing the probabilistic program we are
                 studying, written in language P_0.
             - x: list of variables involved in the program.
             
   2. def compute_statistics(res,xl, e, eps, maxe=1):
          It computes a posteriori statistics for the random variables defined 
          by the considered probabilistic program.
             - res: output samples of the considered probabilistic program.
             - xl: list of variables involved in the program.
             - e: random variable for which we are interested in obtaining 
                  guarantees.
             - eps: width of confidence interval for expectation.
                
        
The main workflow is the following:
    
    #1. define the model
    var('r y i')
    S=seq(draw(r,rhoU()),whl(abs(y)<1,seq(draw(y,rhoG(y,2*r)),setx(i,i+1)),i>=3))
    xlist=['r','y','i']
    
    #2. traslate the model in TF
    tr_S=translate_sc(S_s,xlist) 
    
    #3. create manually a working TF definition, starting from tr_S and using tfd.(...).sample() for vectorial sampling: 
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
    def f(r,y,i,m):
        print("Tracing")
        r = tfd.Uniform(low=r).sample()#draw(r,rho_U())
        def body_f1(r,y,i,m):
            y = tfd.Normal(loc=y, scale=2*r).sample()#draw(y,rho_G(y, 2*r))
            i+=1
            return r,y,i,m
        def body1(r,y,i,m):
            res = tf.where(tf.less(tf.abs(y) ,1.0) & tf.greater(m,0),tf.concat(list(body_f1(r,y,i,m)),axis=0),tf.concat((r,y,i,m),axis=0))
            return  tuple([res[tf.newaxis,j] for j in range(4)]) # slicing tensor res 
        r,y,i,m=tf.while_loop(lambda *_: True, body1, (r,y,i,m), maximum_iterations=100)
        m=tf.where(tf.logical_or(tf.logical_not(tf.less(tf.abs(y) ,1.0)) , tf.equal(m,0.0)),  m * tf.cast(tf.greater_equal(i, 3.0), tf.float32), np.NaN)
        return r,y,i,m
    
    #4. define inputs for the function, and execute the model
    # Tracing
    N=1 
    rr = tf.zeros((1,N))
    yy = tf.zeros((1,N))
    ii = tf.zeros((1,N))
    m = tf.constant(1.0,shape=(1,N))
    res=f(rr,yy,ii,m)  
    # actual execution
    N=10**6
    rr = tf.zeros((1,N))
    yy = tf.zeros((1,N))
    ii = tf.zeros((1,N))
    m = tf.constant(1.0,shape=(1,N))
    res=f(rr,yy,ii,m)
    
    #5. compute posteriori statistics
    [r,y,i]   
    e=r
    eps=0.005
    maxe=1
    exp, lower_prob,conf=compute_statistics(res,xl, e, eps, maxe)
  

Examples and experiments are at the end of the script. 
"""

from  sympy import *
import numpy as np
import time
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


nil=Function('nil')
setx=Function('setx')
draw=Function('draw')
obs=Function('obs')
ite=Function('ite')
whl=Function('whl')

rho=Function('rho')    
rhoU=Function('rhoU')  
rhoG=Function('rhoG')  
B=Function('B')        
G=Function('G')        
g=Function('g') 
K=10
K_str=str(K)
skip=Function('skip')
seq=Function('seq')
c=0



def translate_seq(S,x,ind=''):
    '''
    Given a probabilistic program written in language P_0 described in [1],
    it returns its tanslation into TensorFlow.
    
       - S: model describing the probabilistic program we are
           studying, written in language P_0.
       - x: list of variables involved in the program.
       
    '''
    global c
    xargs=','.join(x)
    f=S.func
    args=S.args
    if f == skip:
        return ""
    elif f == setx:
        xi,g=args
        return f"{ind}{str(xi)}={str(g)}"
    elif f == draw:
        xi,rho=args
        return f"{ind}{str(xi)}=draw({str(rho)})"
    elif f== obs:    
        phi = args
        return f"{ind}m = m * tf.cast({str(phi)},tf.float32)"
    elif f==ite:
        phi, S1, S2 = args
        c=c+1
        c1=str(c)
        c=c+1
        c2=str(c)
        f1 = translate_seq(S1,x,ind+'    ')
        f2 = translate_seq(S2,x,ind+'    ')
        phi_str=str(phi)
        c=c+1
        return ind+f"def f{c1}({xargs},m):\n{f1}\n{ind}    return {xargs},m\n{ind}def f{c2}({xargs},m):\n{f2}\n{ind}    return {xargs},m\n{ind}mask = {phi_str}\n{ind}res=tf.where(mask, tf.concat(f{c1}({xargs},m),axis=0), tf.concat(f{c2}({xargs},m),axis=0))\n{ind}{xargs},m = tuple(res[tf.newaxis,j] for j in range({str(len(x)+1)})) # slicing tensor res"
    elif f==whl:
        phi, S1, psi = args
        phi_str=str(phi)
        psi_str=str(psi)
        c=c+1
        c1=str(c)
        S_tr = translate_seq(S1,x,ind+'    ')
        S_f=       f"def body_f{c1}({xargs},m):\n{S_tr}\n{ind}    return {xargs},m"
        def_body = f"def body{c1}({xargs},m):\n{ind}    res = where(({phi_str}) & tf.greater(m,0.0),tf.concat(body_f{c1}({xargs},m),axis=0),tf.concat(({xargs},m),axis=0))\n{ind}    return tuple([res[tf.newaxis,j] for j in range({str(len(x)+1)})]) # slicing tensor res "
        post=      f"m=tf.where(tf.logical_or(tf.logical_not({phi_str}) , tf.equal(m,0.0)),  m * tf.cast({psi_str},tf.float32), np.NaN)"
        return     f"{ind}{S_f}\n{ind}{def_body}\n{ind}{xargs},m=tf.while_loop(lambda *_: True, body{c1}, ({xargs},m), maximum_iterations={K_str})\n{ind}{post}"
    elif f==seq:
        Slist=[translate_seq(Si,x,ind) for Si in args]           
        return "\n".join(Slist)            
    else:
        print("Syntax error")
        return None


def translate_sc(S,x):
    xargs=','.join(x)
    tr_S = translate_seq(S,x,ind='    ')
    arg_str=','.join(["tf.TensorSpec(shape=None, dtype=tf.float32)"]*(len(x)+1))
    header = f"@tf.function(input_signature=[{arg_str}])"
    return f"{header}\ndef f0({xargs},m):\n{tr_S}\n    return {xargs},m"



def compute_statistics_RW1(res,xl, e, eps, maxe):
    m=res[3][0]
    r=res[0][0]    
    term = (m==1.0)    
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()
    
    LB=r[lt2].numpy().sum()/na-eps     
    UB=(r[term].numpy().sum()+maxe*live.numpy().sum()+10**6*eps)/(term.numpy().sum()-10**6*eps)

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())    
    N=m.shape[0]
    delta2 = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf2 = 1-2*delta2
    exp=[LB,UB]
    return exp, lower_prob, conf2

def compute_statistics_RW2(res,xl, e, eps, maxe):
    m=res[3][0]
    r=res[2][0]>=3
    term = (m==1.0)    
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()
    
    LB=r[lt2].numpy().sum()/na-eps     
    UB=(r[term].numpy().sum()+maxe*live.numpy().sum()+10**6*eps)/(term.numpy().sum()-10**6*eps)

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())    
    N=m.shape[0]
    delta = 2*np.exp(-N*eps**2/maxe**2)+np.exp(-na*eps**2/maxe**2)
    conf = 1-2*delta
    exp=[LB,UB]
    return exp, lower_prob, conf


def compute_statistics_BA(res,xl, e, eps, maxe):
    m=res[4][0]
    N=m.shape[0]

    r=res[1][0]    
    term = (m==1.0)    
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()
    
    LB=r[lt2].numpy().sum()/na-eps   
    UB=r[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta
    exp=[LB,UB]
    return exp, lower_prob, conf

def compute_statistics_TS(res,xl, e, eps, maxe):
    m=res[9][0]
    N=m.shape[0]
    r=res[0][0]    
    term = (m==1.0)    
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)   
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()
    
    LB=r[lt2].numpy().sum()/na-eps     
    UB= r[lt2].numpy().sum()/na+eps   

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta
    exp=[LB,UB]
    return exp, lower_prob, conf

def compute_statistics_CG(res,xl, e, eps, maxe):
    m=res[6][0]
    N=m.shape[0]
    r=res[0][0]    
    term = (m==1.0)    
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)   
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()
    
    LB=r[lt2].numpy().sum()/na-eps    
    UB=r[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta

    exp=[LB,UB]
    return exp, lower_prob, conf

def compute_statistics_MH(res,xl, e, eps, maxe):
    m=res[4][0]
    
    N=m.shape[0]
    r=res[3][0]    
    term = (m==1.0)    
    fail = (m==0.0)      
    live = tf.logical_not(term|fail)   
    lt2=    term|live
    na=live.numpy().sum()+term.numpy().sum()
    
    LB=r[lt2].numpy().sum()/na-eps     
    UB=r[lt2].numpy().sum()/na+eps     

    lower_prob=term.numpy().sum()/(term.numpy().sum()+live.numpy().sum())   
    
    delta = 2*np.exp(-2*N*eps**2/maxe**2)+np.exp(-2*na*eps**2/maxe**2)
    conf = 1-2*delta

    exp=[LB,UB]
    return exp, lower_prob, conf


def compute_expected_value_approx(var, mask):
    return var[mask==1].numpy().sum()/(mask==1).numpy().sum() 




#-------------------------------------------------------------------------------------------
#-------------------------------------  EXPERIMENTS  ------------------------------------
#-------------------------------------------------------------------------------------------

#------------------------------- Example 1: FIRST RANDOM WALK ------------------------------
'''
var('r y i')

S_s=seq(draw(r,rhoU()),whl(abs(y)<1,seq(draw(y,rhoG(y,2*r)),setx(i,i+1)),i>=3))
xlist=['r','y','i']
c=0
#tr_S=translate_sc(S_s,xlist)
#print(tr_S)


#working tf definition
@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(r,y,i,m):
    print("Tracing")
    r = tfd.Uniform(low=r).sample()
    def body_f1(r,y,i,m):
        y = tfd.Normal(loc=y, scale=2*r).sample()
        i+=1
        return r,y,i,m
    def body1(r,y,i,m):
        res = tf.where(tf.less(tf.abs(y) ,1.0) & tf.greater(m,0),tf.concat(list(body_f1(r,y,i,m)),axis=0),tf.concat((r,y,i,m),axis=0))
        return  tuple([res[tf.newaxis,j] for j in range(4)]) 
    r,y,i,m=tf.while_loop(lambda *_: True, body1, (r,y,i,m), maximum_iterations=100)
    m=tf.where(tf.logical_or(tf.logical_not(tf.less(tf.abs(y) ,1.0)) , tf.equal(m,0.0)),  m * tf.cast(tf.greater_equal(i, 3.0), tf.float32), np.NaN)
    return r,y,i,m

var('r y i')

N=1 # Warm up 
rr = tf.zeros((1,N))
yy = tf.zeros((1,N))
ii = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(rr,yy,ii,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)

N=10**6  
rr = tf.zeros((1,N))
yy = tf.zeros((1,N))
ii = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f0(rr,yy,ii,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M elems  %s seconds -------        " % final_time)


xl=[r,y,i]   
e=r
eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_RW1(res,xl, e, eps, maxe)
'''



#------------------------------- Example 2: SECOND RANDOM WALK ------------------------------

'''
var('r y i')

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f01(r,y,i,m):
    print("Tracing")
    r = tfd.Uniform(low=r).sample()
    def body_f1(y,i):
        y = tfd.Normal(loc=y, scale=2*r).sample()
        i+=1.0
        return y,i
    def body1(y,i):
        res = tf.where(tf.less(tf.abs(y) ,1.0) ,tf.concat((body_f1(y,i)),axis=0),tf.concat((y,i),axis=0))
        return  res[tf.newaxis,0],res[tf.newaxis,1]
    y,i=tf.while_loop(lambda *_: True, body1, (y,i), maximum_iterations=5)
    m=tf.where(tf.greater_equal(tf.abs(y) ,1.0) ,   m*tf.cast(True,tf.float32), np.NaN)
    return r,y,i,m



N=1 # Warm up (Tracing)
rr = tf.zeros((1,N))
yy = tf.zeros((1,N))
ii = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f01(rr,yy,ii,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1  elem %s seconds -------        " % final_time)

N=10**6 # actual execution
rr = tf.zeros((1,N))
yy = tf.zeros((1,N))
ii = tf.zeros((1,N))
m = tf.constant(1.0,shape=(1,N))
start_time=time.time()
res=f01(rr,yy,ii,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 10**6 elems  %s seconds -------        " % final_time)


xl=[r,y,i]    
e=i>=3
eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_RW2(res,xl, e, eps, maxe)
'''

#--------------------------------------- Example 3: Burglar Alarm -----------------------------
'''
var('earthquake burglary phoneWorking maryWakes alarm called')
BA = seq(draw(earthquake,B(0.001)), 
         draw(burglary,B(0.01)) ,   
         setx(alarm , (earthquake>0) | (burglary>0)),
         ite(earthquake>0, draw(phoneWorking,B(0.6)), draw(phoneWorking,B(0.99))),
         ite(alarm & (earthquake>0), 
             draw(maryWakes,B(0.8)), 
             ite(alarm, 
                 draw(maryWakes,B(0.6)), 
                 draw(maryWakes,B(0.2))
                )
            ),
         setx(called , (maryWakes>0) & (phoneWorking>0)),
         obs(called)
)

xlist_BA='earthquake, burglary, phoneWorking, maryWakes'.split(',')
c=0
#tr_BA=translate_sc(BA,xlist_BA)
#print(tr_BA)



# working TF function definition:
@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)]+[tf.TensorSpec(shape=None, dtype=tf.float32)]*7)
def f0(earthquake, burglary, phoneWorking, maryWakes,m,p1,p2,p3,p4,p5,p6,p7):
    print("Tracing")
    earthquake=tfd.Bernoulli(dtype=tf.float32, probs=p1).sample()
    burglary=tfd.Bernoulli(dtype=tf.float32,probs=p2).sample()
    alarm=(burglary > 0) | (earthquake > 0)
    def f1(earthquake, burglary, phoneWorking, maryWakes,m):
        phoneWorking=tfd.Bernoulli(dtype=tf.float32,probs=p3).sample()
        return earthquake, burglary, phoneWorking, maryWakes,m
    def f2(earthquake, burglary, phoneWorking, maryWakes,m):
        phoneWorking=tfd.Bernoulli(dtype=tf.float32,probs=p4).sample()
        return earthquake, burglary, phoneWorking, maryWakes,m
    mask = earthquake > 0
    res=tf.where(mask, tf.concat(f1(earthquake, burglary, phoneWorking, maryWakes,m),axis=0), tf.concat(f2(earthquake, burglary, phoneWorking, maryWakes,m),axis=0))
    earthquake, burglary, phoneWorking, maryWakes,m = tuple(res[tf.newaxis,j] for j in range(5))
    def f4(earthquake, burglary, phoneWorking, maryWakes,m):
        maryWakes=tfd.Bernoulli(dtype=tf.float32,probs=p5).sample()
        return earthquake, burglary, phoneWorking, maryWakes,m
    def f5(earthquake, burglary, phoneWorking, maryWakes,m):
        def f6(earthquake, burglary, phoneWorking, maryWakes,m):
            maryWakes=tfd.Bernoulli(dtype=tf.float32,probs=p6).sample()
            return earthquake, burglary, phoneWorking, maryWakes,m
        def f7(earthquake, burglary, phoneWorking, maryWakes,m):
            maryWakes=tfd.Bernoulli(dtype=tf.float32,probs=p7).sample()
            return earthquake, burglary, phoneWorking, maryWakes,m
        mask = alarm
        res=tf.where(mask, tf.concat(f6(earthquake, burglary, phoneWorking, maryWakes,m),axis=0), tf.concat(f7(earthquake, burglary, phoneWorking, maryWakes,m),axis=0))
        earthquake, burglary, phoneWorking, maryWakes,m = tuple(res[tf.newaxis,j] for j in range(5)) 
        return earthquake, burglary, phoneWorking, maryWakes,m
    mask = alarm & (earthquake > 0)
    res=tf.where(mask, tf.concat(f4(earthquake, burglary, phoneWorking, maryWakes,m),axis=0), tf.concat(f5(earthquake, burglary, phoneWorking, maryWakes,m),axis=0))
    earthquake, burglary, phoneWorking, maryWakes,m = tuple(res[tf.newaxis,j] for j in range(5))
    called=(maryWakes > 0) & (phoneWorking > 0)
    m = m * tf.cast((called),tf.float32)
    return earthquake, burglary, phoneWorking, maryWakes,m



# Warm up
N=1
bb=tf.zeros(shape=(1,N))
m=tf.fill(dims=[1,N],value=1.0)
shp=(1,N)
p1=tf.constant(.001 , shape=shp)
p2=tf.constant(.01 , shape=shp)
p3=tf.constant(.6 , shape=shp)
p4=tf.constant(.99 , shape=shp)
p5=tf.constant(.8 , shape=shp)
p6=tf.constant(.6 , shape=shp)
p7=tf.constant(.2 , shape=shp)
start_time=time.time()
res=f0(bb,bb,bb,bb,m,p1,p2,p3,p4,p5,p6,p7)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1 elems  %s seconds -------        " % final_time)


N=10**6
bb=tf.zeros(shape=(1,N))
m=tf.fill(dims=[1,N],value=1.0)
shp=(1,N)
p1=tf.constant(.001 , shape=shp)
p2=tf.constant(.01 , shape=shp)
p3=tf.constant(.6 , shape=shp)
p4=tf.constant(.99 , shape=shp)
p5=tf.constant(.8 , shape=shp)
p6=tf.constant(.6 , shape=shp)
p7=tf.constant(.2 , shape=shp)

start_time=time.time()
res=f0(bb,bb,bb,bb,m,p1,p2,p3,p4,p5,p6,p7)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 10**6 elems  %s seconds -------        " % final_time)

compute_expected_value_approx(res[1][0], res[4][0])

var('earthquake, burglary, phoneWorking, maryWakes')
xl=[earthquake, burglary, phoneWorking, maryWakes]   
e=burglary
eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_BA(res,xl, e, eps, maxe)
'''

#--------------------------------------- Example 4: TrueSkill -----------------------------
'''
scale=100
var('skillA skillB skillC perfA1 perfB1 perfB2 perfC2 perfA3 perfC3')
TS = seq(draw(skillA,G(100/scale,10/scale)), 
         draw(skillB,G(100/scale,10/scale)),
         draw(skillC,G(100/scale,10/scale)),
         
         draw(perfA1, G(skillA,15/scale)),
         draw(perfB1, G(skillB,15/scale)),
         
         obs(perfA1 > perfB1),
         draw(perfB2, G(skillB,15/scale)),
         draw(perfC2, G(skillC,15/scale)),
         
         obs(perfB2 > perfC2),
         draw(perfA3, G(skillA,15/scale)),
         draw(perfC3, G(skillC,15/scale)),
         obs(perfA3 > perfC3)
         )

xlist_TS='skillA, skillB, skillC, perfA1, perfB1, perfB2, perfC2, perfA3, perfC3'.split(',')
c=0
#tr_TS=translate_sc(TS,xlist_TS)
#print(tr_TS)

# working TF function definition
@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)]+[tf.TensorSpec(shape=None, dtype=tf.float32)]*3)
def f0(skillA, skillB, skillC, perfA1, perfB1, perfB2, perfC2, perfA3, perfC3,m,p1,p2,p3):
    print("Tracing")
    skillA=tfd.Normal(loc=p1, scale=p2).sample() 
    skillB=tfd.Normal(loc=p1, scale=p2).sample()
    skillC=tfd.Normal(loc=p1, scale=p2).sample()
    perfA1=tfd.Normal(loc=skillA, scale=p3).sample()
    perfB1=tfd.Normal(loc=skillB, scale=p3).sample()
    m = m * tf.cast((perfA1-perfB1>0),tf.float32)
    perfB2=tfd.Normal(loc=skillB, scale=p3).sample()
    perfC2=tfd.Normal(loc=skillC, scale=p3).sample()
    m = m * tf.cast((perfB2-perfC2>0),tf.float32)
    perfA3=tfd.Normal(loc=skillA, scale=p3).sample()
    perfC3=tfd.Normal(loc=skillC, scale=p3).sample()
    m = m * tf.cast((perfA3-perfC3>0),tf.float32)
    return skillA, skillB, skillC, perfA1, perfB1, perfB2, perfC2, perfA3, perfC3,m


# Warm up;
N=1
bb=tf.zeros(shape=(1,N))
m=tf.fill(dims=[1,N],value=1.0)
shp=(1,N)
p1=1+tf.zeros(shape=(1,N))
p2=0.1+tf.zeros(shape=(1,N))
p3=0.15+tf.zeros(shape=(1,N))
start_time=time.time()
res=f0(bb,bb,bb,bb,bb,bb,bb,bb,bb,m,p1,p2,p3)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1 elems  %s seconds -------        " % final_time)

# actual execution:
N=10**6
bb=tf.zeros(shape=(1,N))
m=tf.fill(dims=[1,N],value=1.0)
shp=(1,N)
p1=1+tf.zeros(shape=(1,N))
p2=0.1+tf.zeros(shape=(1,N))
p3=0.15+tf.zeros(shape=(1,N))
start_time=time.time()
res=f0(bb,bb,bb,bb,bb,bb,bb,bb,bb,m,p1,p2,p3)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 10**6 elems  %s seconds -------        " % final_time)

compute_expected_value_approx(res[0], res[9])


var('skillA, skillB, skillC, perfA1, perfB1, perfB2, perfC2, perfA3, perfC3')
xl=[skillA, skillB, skillC, perfA1, perfB1, perfB2, perfC2, perfA3, perfC3]   
e=skillA
eps=0.005
maxe=1 #0.6
exp, lower_prob,conf=compute_statistics_TS(res,xl, e, eps, maxe)
'''

#--------------------------------------- Example 5: ClickGraph  ----------------------------

'''
var('simAll, sim, p1, p2, clickA, clickB')
ClickGraph = seq(
                draw(simAll, rhoU(0,1)),
                
                draw(sim, B(simAll)),  
                draw(p1, rhoU(0,1)),
                ite(sim>0, setx(p2,p1), 
                           draw(p2, rhoU(0,1))),
                draw(clickA, B(p1)),
                draw(clickB, B(p2)),
                obs(clickA>0),
                obs(clickB>0),           
                
                draw(sim, B(simAll)),  
                draw(p1, rhoU(0,1)),
                ite(sim>0, setx(p2,p1), 
                           draw(p2, rhoU(0,1))),
                draw(clickA, B(p1)), 
                draw(clickB, B(p2)), 
                obs(clickA>0),
                obs(clickB>0),
                
                draw(sim, B(simAll)),  
                draw(p1, rhoU(0,1)),
                ite(sim>0, setx(p2,p1), 
                           draw(p2, rhoU(0,1))),
                draw(clickA, B(p1)),
                draw(clickB, B(p2)),
                obs(clickA>0),
                obs(clickB>0),   
                
                
                draw(sim, B(simAll)),  
                draw(p1, rhoU(0,1)),
                ite(sim>0, setx(p2,p1), 
                           draw(p2, rhoU(0,1))),
                draw(clickA, B(p1)),
                draw(clickB, B(p2)),
                obs(clickA==0),
                obs(clickB==0),           
                
                
                draw(sim, B(simAll)),  
                draw(p1, rhoU(0,1)),
                ite(sim>0, setx(p2,p1), 
                           draw(p2, rhoU(0,1))),
                draw(clickA, B(p1)),
                draw(clickB, B(p2)),
                obs(clickA==0),
                obs(clickB==0)           
                )


xlist_ClickGraph='simAll, sim, p1, p2, clickA, clickB '.split(',')
c=0
#tr_CG=translate_sc(ClickGraph,xlist_ClickGraph)
#print(tr_CG)

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]*8)  
def f0(r,simAll, sim, p1, p2, clickA, clickB ,m):
    print("Tracing")
    simAll=tfd.Uniform(low=r).sample()
 
    sim=tfd.Bernoulli(dtype=tf.float32,probs=simAll).sample()
    p1=tfd.Uniform(low=r).sample()
    def f1(simAll, sim, p1, p2, clickA, clickB ,m):
        p2=p1
        return simAll, sim, p1, p2, clickA, clickB ,m
    def f2(simAll, sim, p1, p2, clickA, clickB ,m):
        p2=tfd.Uniform(low=r).sample()
        return simAll, sim, p1, p2, clickA, clickB ,m
    mask = sim > 0
    res=tf.where(mask, tf.concat(f1(simAll, sim, p1, p2, clickA, clickB,m),axis=0), tf.concat(f2(simAll, sim, p1, p2, clickA, clickB,m),axis=0))  
    simAll, sim, p1, p2, clickA, clickB ,m = tuple(res[tf.newaxis,j] for j in range(7)) 
    clickA=tfd.Bernoulli(dtype=tf.float32,probs=p1).sample()
    clickB=tfd.Bernoulli(dtype=tf.float32,probs=p2).sample()
    m = m * tf.cast((clickA > 0),tf.float32)
    m = m * tf.cast((clickB > 0),tf.float32)
    
    
    sim=tfd.Bernoulli(dtype=tf.float32,probs=simAll).sample()
    p1=tfd.Uniform(low=r).sample()
    def f4(simAll, sim, p1, p2, clickA, clickB ,m):
        p2=p1
        return simAll, sim, p1, p2, clickA, clickB ,m
    def f5(simAll, sim, p1, p2, clickA, clickB ,m):
        p2=tfd.Uniform(low=r).sample()
        return simAll, sim, p1, p2, clickA, clickB ,m
    mask = sim > 0
    res=tf.where(mask, tf.concat(f4(simAll, sim, p1, p2, clickA, clickB ,m),axis=0), tf.concat(f5(simAll, sim, p1, p2, clickA, clickB ,m),axis=0))
    simAll, sim, p1, p2, clickA, clickB ,m = tuple(res[tf.newaxis,j] for j in range(7))
    clickA=tfd.Bernoulli(dtype=tf.float32,probs=p1).sample()
    clickB=tfd.Bernoulli(dtype=tf.float32,probs=p2).sample()
    m = m * tf.cast((clickA > 0),tf.float32)
    m = m * tf.cast((clickB > 0),tf.float32)
        
    sim=tfd.Bernoulli(dtype=tf.float32,probs=simAll).sample()
    p1=tfd.Uniform(low=r).sample()
    def f1(simAll, sim, p1, p2, clickA, clickB ,m):
        p2=p1
        return simAll, sim, p1, p2, clickA, clickB ,m
    def f2(simAll, sim, p1, p2, clickA, clickB ,m):
        p2=tfd.Uniform(low=r).sample()
        return simAll, sim, p1, p2, clickA, clickB ,m
    mask = sim > 0
    res=tf.where(mask, tf.concat(f1(simAll, sim, p1, p2, clickA, clickB,m),axis=0), tf.concat(f2(simAll, sim, p1, p2, clickA, clickB,m),axis=0))  
    simAll, sim, p1, p2, clickA, clickB ,m = tuple(res[tf.newaxis,j] for j in range(7))
    clickA=tfd.Bernoulli(dtype=tf.float32,probs=p1).sample()
    clickB=tfd.Bernoulli(dtype=tf.float32,probs=p2).sample()
    m = m * tf.cast((clickA < 1),tf.float32)
    m = m * tf.cast((clickB < 1),tf.float32)
    return simAll, sim, p1, p2, clickA, clickB ,m

N=1
rr = tf.zeros((1,N))
bb=tf.zeros(shape=(1,N))
m=tf.fill(dims=[1,N],value=1.0)
start_time=time.time()
res=f0(rr,bb,bb,bb,bb,bb,bb,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1 elems  %s seconds -------        " % final_time)

N=10**6
rr = tf.zeros((1,N))
bb=tf.zeros(shape=(1,N))
m=tf.fill(dims=[1,N],value=1.0)
start_time=time.time()
res=f0(rr,bb,bb,bb,bb,bb,bb,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 10*6 elems  %s seconds -------        " % final_time)


var('simAll, sim, p1, p2, clickA, clickB ,m')
xl=[simAll, sim, p1, p2, clickA, clickB]   
e=simAll
eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_CG(res,xl, e, eps, maxe)
'''

#---------------------------------- Example 6: Monty Hall  ----------------------------
'''
var('x car guest host guest2 win')

def ternary(x,pL=[1/3,1/3,1/3],vL=[0,1,2]):   
    pL=np.cumsum(pL) 
    S=seq( draw(x,rhoU()), ite(x<=pL[0],setx(x,vL[0]), ite(x<=pL[1],setx(x,vL[1]),setx(x,vL[2]))))
    return S

def bern(x,e=1/2,vL=[0,1]):  
    S=seq(draw(x,rhoU()),ite(x>=e,setx(x,vL[0]),setx(x,vL[1])))
    return S  
    
MH = seq(ternary(car),
                  ternary(guest),      
                  ite(Eq(car,0) & Eq(guest,1), setx(host,2) ,setx(host,host)),
                  ite(Eq(car,0) & Eq(guest,2), setx(host,1),setx(host,host)),                 
                  ite(Eq(car,1) & Eq(guest,0),setx(host,2),setx(host,host)),
                  ite(Eq(car,1) & Eq(guest,2),setx(host,0),setx(host,host)),
                  ite(Eq(car,2) & Eq(guest,0), setx(host,1),setx(host,host)),
                  ite(Eq(car,2) & Eq(guest,1), setx(host,0),setx(host,host)),
                  ite(Eq(car,0) & Eq(guest,0), seq(draw(x,rhoU()),ite(x>=0.5,setx(host,1),setx(host,2))),setx(host,host)),
                  ite(Eq(car,1) & Eq(guest,1), seq(draw(x,rhoU()),ite(x>=0.5,setx(host,0),setx(host,2))),setx(host,host)),
                  ite(Eq(car,2) & Eq(guest,2), seq(draw(x,rhoU()),ite(x>=0.5,setx(host,0),setx(host,1))),setx(host,host)),
                  ite(Eq(guest,1) & Eq(host,2),setx(win,0),setx(win,win)),
                  ite(Eq(guest,0) & Eq(host,2),setx(guest,1),setx(win,win)),
                  ite(Eq(guest,0) & Eq(host,1),setx(win,2),setx(win,win)),
                  ite(Eq(guest,2) & Eq(host,1),setx(win,0),setx(win,win)),                  
                  ite(Eq(guest,1) & Eq(host,0),setx(win,2),setx(win,win)),
                  ite(Eq(guest,2) & Eq(host,0),setx(win,1),setx(win,win)),                  
                  ite(Eq(win,car),
                       setx(win,1),
                       setx(win,0)))

xlist_MH='car, guest, host, win'.split(',')
c=0

@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32),tf.TensorSpec(shape=None, dtype=tf.float32)])
def f0(r, car, guest, host, win,m):
    car=tfd.Uniform(low=r).sample()
    def f1(car, guest, host, win,m):
        car=r
        return car, guest, host, win,m
    def f2(car, guest, host, win,m):
        def f3(car, guest, host, win,m):
            car=r+1
            return car, guest, host, win,m
        def f4(car, guest, host, win,m):
            car=r+2
            return car, guest, host, win,m
        mask = car <= 0.666666666666667
        res=tf.where(mask, tf.concat(f3(car, guest, host, win,m),axis=0), tf.concat(f4(car, guest, host, win,m),axis=0))
        car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
        return car, guest, host, win,m
    mask = car <= 0.333333333333333
    res=tf.where(mask, tf.concat(f1(car, guest, host, win,m),axis=0), tf.concat(f2(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res

   
    guest=tfd.Uniform(low=r).sample()
    def f7(car, guest, host, win,m):
        guest=r
        return car, guest, host, win,m
    def f8(car, guest, host, win,m):
        def f9(car, guest, host, win,m):
            guest=r+1
            return car, guest, host, win,m
        def f10(car, guest, host, win,m):
            guest=r+2
            return car, guest, host, win,m
        mask = guest <= 0.666666666666667
        res=tf.where(mask, tf.concat(f9(car, guest, host, win,m),axis=0), tf.concat(f10(car, guest, host, win,m),axis=0))
        car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
        return car, guest, host, win,m
    mask = guest <= 0.333333333333333
    res=tf.where(mask, tf.concat(f7(car, guest, host, win,m),axis=0), tf.concat(f8(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
    def f13(car, guest, host, win,m):
        host=r+2
        return car, guest, host, win,m
    def f14(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    
    mask1 = (car==r)
    mask2 = (guest==r+1)
    #mask = (car==r) and (guest==r+1)
    res=tf.where(mask1 & mask2, tf.concat(f13(car, guest, host, win,m),axis=0), tf.concat(f14(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
   
    def f16(car, guest, host, win,m):
        host=r+1
        return car, guest, host, win,m
    def f17(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    mask1 = (car==r)
    mask2 = (guest==r+2)
    res=tf.where(mask1 & mask2, tf.concat(f16(car, guest, host, win,m),axis=0), tf.concat(f17(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
   
    def f19(car, guest, host, win,m):
        host=r+2
        return car, guest, host, win,m
    def f20(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    mask1 = (car==r+1)
    mask2 = (guest==r)
    res=tf.where(mask1 & mask2, tf.concat(f19(car, guest, host, win,m),axis=0), tf.concat(f20(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
   
    def f22(car, guest, host, win,m):
        host=r
        return car, guest, host, win,m
    def f23(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    mask1 = (car==r+1)
    mask2 = (guest==r+2)
    res=tf.where(mask1 & mask2, tf.concat(f22(car, guest, host, win,m),axis=0), tf.concat(f23(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
      
    def f25(car, guest, host, win,m):
        host=r+1
        return car, guest, host, win,m
    def f26(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m    
    mask1 = (car==r+2)
    mask2 = (guest==r)
    res=tf.where(mask1 & mask2, tf.concat(f25(car, guest, host, win,m),axis=0), tf.concat(f26(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
    def f28(car, guest, host, win,m):
        host=r+0
        return car, guest, host, win,m
    def f29(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    mask1 = (car==r+2)
    mask2 = (guest==r+1)
    res=tf.where(mask1 & mask2, tf.concat(f28(car, guest, host, win,m),axis=0), tf.concat(f29(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res

    
    def f31(car, guest, host, win,m):
        x=tfd.Uniform(low=r).sample()
        def f33(car, guest, host, win,m):
            host=r+1
            return car, guest, host, win,m
        def f34(car, guest, host, win,m):
            host=r+2
            return car, guest, host, win,m
        mask = x >= 0.5
        res=tf.where(mask, tf.concat(f33(car, guest, host, win,m),axis=0), tf.concat(f34(car, guest, host, win,m),axis=0))
        car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
        return car, guest, host, win,m
    def f32(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    mask1 = (car==r)
    mask2 = (guest==r)
    res=tf.where(mask1 & mask2, tf.concat(f31(car, guest, host, win,m),axis=0), tf.concat(f32(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
    def f37(car, guest, host, win,m):
        x=tfd.Uniform(low=r).sample()
        def f39(car, guest, host, win,m):
            host=r
            return car, guest, host, win,m
        def f40(car, guest, host, win,m):
            host=r+2
            return car, guest, host, win,m
        mask = x >= 0.5
        res=tf.where(mask, tf.concat(f39(car, guest, host, win,m),axis=0), tf.concat(f40(car, guest, host, win,m),axis=0))
        car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
        return car, guest, host, win,m
    def f38(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    mask1 = (car==r+1)
    mask2 = (guest==r+1)
    res=tf.where(mask1 & mask2, tf.concat(f37(car, guest, host, win,m),axis=0), tf.concat(f38(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
    def f43(car, guest, host, win,m):
        x=tfd.Uniform(low=r).sample()
        def f45(car, guest, host, win,m):
            host=r+0
            return car, guest, host, win,m
        def f46(car, guest, host, win,m):
            host=r+1
            return car, guest, host, win,m
        mask = x >= 0.5
        res=tf.where(mask, tf.concat(f45(car, guest, host, win,m),axis=0), tf.concat(f46(car, guest, host, win,m),axis=0))
        car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
        return car, guest, host, win,m
    def f44(car, guest, host, win,m):
        host=host
        return car, guest, host, win,m
    mask1 = (car==r+2)
    mask2 = (guest==r+2)
    res=tf.where(mask1 & mask2, tf.concat(f43(car, guest, host, win,m),axis=0), tf.concat(f44(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
    #-------------  
    def f49(car, guest, host, win,m):
        win=r
        return car, guest, host, win,m
    def f50(car, guest, host, win,m):
        win=win
        return car, guest, host, win,m
    mask1 = (guest==r+1)
    mask2 = (host==r+2)
    res=tf.where(mask1 & mask2, tf.concat(f49(car, guest, host, win,m),axis=0), tf.concat(f50(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    def f52(car, guest, host, win,m):
        win=r+1
        return car, guest, host, win,m
    def f53(car, guest, host, win,m):
        win=win
        return car, guest, host, win,m
    mask1 = (guest==r)
    mask2 = (host==r+2)
    res=tf.where(mask1 & mask2, tf.concat(f52(car, guest, host, win,m),axis=0), tf.concat(f53(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    def f55(car, guest, host, win,m):
        win=r+2
        return car, guest, host, win,m
    def f56(car, guest, host, win,m):
        win=win
        return car, guest, host, win,m
    mask1 = (guest==r)
    mask2 = (host==r+1)
    res=tf.where(mask1 & mask2, tf.concat(f55(car, guest, host, win,m),axis=0), tf.concat(f56(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    def f58(car, guest, host, win,m):
        win=r
        return car, guest, host, win,m
    def f59(car, guest, host, win,m):
        win=win
        return car, guest, host, win,m
    mask1 = (guest==r+2)
    mask2 = (host==r+1)   
    res=tf.where(mask1 & mask2, tf.concat(f58(car, guest, host, win,m),axis=0), tf.concat(f59(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    def f61(car, guest, host, win,m):
        win=r+2
        return car, guest, host, win,m
    def f62(car, guest, host, win,m):
        win=win
        return car, guest, host, win,m
    mask1 = (guest==r+1)
    mask2 = (host==r)
    res=tf.where(mask1 & mask2, tf.concat(f61(car, guest, host, win,m),axis=0), tf.concat(f62(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    def f64(car, guest, host, win,m):
        win=r+1
        return car, guest, host, win,m
    def f65(car, guest, host, win,m):
        win=win
        return car, guest, host, win,m
    mask1 = (guest==r+2)
    mask2 = (host==r)
    res=tf.where(mask1 & mask2, tf.concat(f64(car, guest, host, win,m),axis=0), tf.concat(f65(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res
    
    
    def f67(car, guest, host, win,m):
        win=r+1
        return car, guest, host, win,m
    def f68(car, guest, host, win,m):
        win=r
        return car, guest, host, win,m
    mask = win==car
    res=tf.where(mask, tf.concat(f67(car, guest, host, win,m),axis=0), tf.concat(f68(car, guest, host, win,m),axis=0))
    car, guest, host, win,m = tuple(res[tf.newaxis,j] for j in range(5)) # slicing tensor res

    return car, guest, host, win,m



N=1
rr = tf.zeros((1,N))
bb=tf.zeros(shape=(1,N))
m=tf.fill(dims=[1,N],value=1.0)
start_time=time.time()
res=f0(rr,bb,bb,bb,bb,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1 elems  %s seconds ------      " % final_time)



N=10**6
rr = tf.zeros((1,N))
bb=tf.zeros(shape=(1,N))
m=tf.fill(dims=[1,N],value=1.0)
start_time=time.time()
res=f0(rr,bb,bb,bb,bb,m)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 10**6 elems  %s seconds -------        " % final_time)



var('car guest host guest2 win')
xl=[car,guest,host, win]   
e=win
eps=0.005
maxe=1
exp, lower_prob,conf=compute_statistics_MH(res,xl, e, eps, maxe)
'''
