"""
Python proof-of-concept SIMD-parallel implementation of A,
an algorithm to compute guaranteed estimates for the expectation
of random variables defined by probabilistic programs. It exploits 
Importance Sampling to limit the rejections problem and is based 
on a small-step semantics for probabilistic programming 
described in: 'Guaranteed inference for probabilistic programs: 
a small-step operational approach',  the guaratees are given in 
the form of confidence intervals.


The main function is:  
    
    - def semPP_3_r(P,e,delta=0.001,N=100):
          It returns the exact confidence interval of e with a given confidence eps:
             - P: model describing the probabilistic program we are
                 studying.
             - e: random variable for which we are interested in obtaining 
                  guarantees on the expected value.
             - delta: small positive number, s.t. eps= 1-delta is the
                      confidence of the interval returned by the function.
             - N: number of samples for MC estimation.
            
Examples and experiments are at the end of the script. 


"""

import sympy
from sympy.utilities.iterables import flatten
from sympy import prod
import numpy as np
from sympy import init_printing
init_printing(use_latex=False)
from sympy import *
import time
from mpmath import iv
import copy
from numpy import random   
from scipy import stats
from scipy.stats import truncnorm


verbose=False
def printv(*s):
    global verbose
    if verbose:
        print(*s)
        
        
U = Function('U')  
G = Function('G')  
var('x y u v w u1 u2 u3 u4 u5 u6 u7')
support_dic = {U:[0,1],G:[-np.inf,np.inf]}
range_dic = {U:1,G:1}
sampler_dic={'U': random.uniform(), 'G':lambda _,mu,sc,ll,lr: truncnorm.rvs(a=(ll - mu) /sc,b=(lr - mu) /sc,loc=mu,scale=sc) }            
drawTriple=Function('drwTr')
Condition =Function('cond')
wn=Function('wn')

######## syntax of PP #################
One=sympify(1)
tt=sympify(True)
skip=Function('skip')
setv =Function('setv') 
drawv=Function('drawv') 
ifte =Function('ifte') 
obs  =Function('obs') 
seq= Function('seq')
rec= Function('rec')
whl= Function('whl')


varcount=0
def newvar(name='u'):
    global varcount
    w=sympy.var(name+'_'+str(varcount) ,real=True)    
    varcount+=1
    return w

def termination(S):
    op=S.func
    if op==whl or op==ifte or op==drawv or op==setv or op==obs:
        return False
    if op==skip:
        return True
    if op==seq:
        pList=S.args
        return all([termination(p) for p in pList])
    

def collectS_s(e,drawList,term,S,k=10,simp=False):  # collecting semantics of statements S; e=expression in the program variables (represents f); psi=logical formula in the program variables
    if k==0:
        return [(e,drawList,termination(S))]
    op=S.func

    if op==skip:
        return [ (e,drawList,True)]
    if op==setv:
        xi= S.args[0]
        g=  S.args[1]
        return [ (e.subs({xi:g}),[drt.subs({xi:g}) for drt in drawList ],term) ]
    if op==drawv:
        xi=  S.args[0]
        rho= S.args[1]
        u= newvar('u')
        return [ (e.subs({xi:u}), [drawTriple(u,True,rho)]+[drt.subs({xi:u}) for drt in drawList ],term ) ]
    if op==obs:
        phi=S.args[0]
        if simp:
            phi=simplify_logic(phi)
        return [ (e, [Condition(phi)]+drawList, term)]
    if op==ifte:
        phi=S.args[0]
        S1 =S.args[1]
        S2 =S.args[2]
        resS1=collectS_s(e,drawList,term,S1,k=k,simp=simp)
        thenRes=[ (e0,  [Condition(phi)]+drL,trm) for e0,drL,trm in resS1 ]
        resS2=collectS_s(e,drawList,term,S2,k=k,simp=simp)
        thenElse=[ (e0,  [Condition(Not(phi))]+drL,trm) for e0,drL,trm in resS2 ]
        return thenRes+thenElse
    if op==seq:
        SL =list(S.args)[::-1] 
        coll=[(e,drawList,term)]
        for S0 in SL:
            newcoll=[]
            for (e0,drL,trm) in coll:
                newcoll=newcoll+collectS_s(e0,drL,trm,S0,k=k,simp=simp)
            coll=newcoll
        return coll
    if op==whl:
        phi=S.args[0]
        S0= S.args[1]
        return collectS_s(e,drawList,term,ifte(phi,seq(S0,S),skip()),k=k-1,simp=simp)
    print("Syntax error")
    return None 



def preprocessBranch(drawList,simp=False):
    varorder=[]
    indicesDL={}
    for e,j in zip(drawList,range(len(drawList))):
        if e.func==drawTriple:
            x=e.args[0]
            varorder.append(x)
            indicesDL[x]=j
    for e,i in zip(drawList,range(len(drawList))):
        if e.func==Condition:
            phi=e.args[0]
            if phi==False:
                return [], []
            if phi==True:
                continue
            varphi=phi.free_symbols
            xmax=varorder[max([varorder.index(x) for x in varphi])]
            j=indicesDL[xmax]
            tr_args=drawList[j].args
            if simp:
                drawList[j]=drawTriple(xmax, simplify_logic(tr_args[1] &  phi), tr_args[2])
            else:
                drawList[j]=drawTriple(xmax,tr_args[1] &  phi, tr_args[2])
            drawList[i]=Condition(True)
    drawList=[ e for e in drawList if e!=Condition(True)]
    return drawList, varorder




################ for (optional) static bound analysis ################
def int2iv(R):
    '''
    Returns a representation of rectangle R as a list of intervals in the mpmath Python library.
    '''
    return [iv.mpf(ii) for ii in R]

import numbers
def iv2intv(Iv):
    '''
    Inverts rectangle representation from mpmath to Python
    '''
    i=0
    for z in Iv:
        if isinstance(z, numbers.Number): 
            z=iv.mpf(z)
            Iv[i]=z
        i=i+1
    return [ [float(z.a),float(z.b)] for z in Iv ]

def findBoundW_A(dL,varorder,e=One,baseIntv=[0,1],truncL=-np.inf,truncR=np.inf): #finds static bound on variation of e(x)*weight(x)
    k=len(varorder)
    rangevar=[baseIntv]*k
    minL=1
    maxU=1
    upperW=1
    lowerW=1
    for el in dL:
        u=el.args[0]
        phi=el.args[1]
        rho=el.args[2]
        f=rho.func
        LL,UL=solveIneqSymb(phi,u,f)
        rho1=f(*(rho.args+(LL,UL)))            
        if f==U:
            ll,rl=rho1.args[0],rho1.args[1]
            idx=varorder.index(u)
            upperW=upperW*rl
            lowerW=lowerW*ll
            rangevar[idx]=[0,1]
            if  sympify(rl).is_constant():
                rangevar[idx][1]=float(rl)
                maxU=maxU*rl
            if  sympify(ll).is_constant():
                rangevar[idx][0]=float(ll)
                minL=minL*ll
            else:
                minL=0
        elif f==G:
            loc,sc,ll,rl=rho1.args[0],rho1.args[1],rho1.args[2],rho1.args[3]
            idx=varorder.index(u)
            if (G(loc,sc,ll,rl)).is_constant():
               w=stats.norm.cdf(float(rl),loc=float(loc),scale=float(sc))-stats.norm.cdf(float(ll),loc=float(loc),scale=float(sc))
               upperW=upperW*w
               lowerW=lowerW*w
               minL=minL*w
               maxU=maxU*w
               rangevar[idx]=[float(ll),float(rl)]
            else:
                lowerW=0
                minL=0
                rangevar[idx]=[truncL,truncR]#[stats.norm.ppf(deltatail,loc=loc,scale=sc),stats.norm.ppf(1-deltatail,loc=loc,scale=sc)]
        else:
            print("Distribution not supported")
            return None
    we=e*upperW-e*lowerW
    if we.is_constant():
        return we
    fwe=lambdify(varorder,we,modules=iv)
    res=iv2intv([fwe(*int2iv(rangevar))])[0]
    Up=res[1]
    Lo=res[0]    
    Lo=max(minL*truncL,Lo)
    Up=min(maxU*truncR,float(Up))
    return Up-Lo       

######## Distributions  #####################
var('z')
log_cdfG=  lambda*x: np.log(stats.norm.cdf(x=x[3],loc=x[0],scale=x[1])-stats.norm.cdf(x=x[2],loc=x[0],scale=x[1]))
trunc_norm=lambda*x: truncnorm.rvs(a=(x[2]-x[0])/x[1],b=(x[3]-x[0])/x[1],loc=x[0],scale=x[1])
uniform =  lambda*x: np.random.uniform(low=x[0],high=x[1])
log_cdfU=  lambda*x: np.log(x[1]-x[0])

NKG=iv.sqrt(2*iv.pi) # Default normalization constant for Gaussian distributions
KG = sqrt(2*pi)
def gauss(x,mu,var):
    return gauss_u(x,mu,var)/NKG

mymathmodule=[{'log':np.log,'exp':np.exp, 'sqrt':np.sqrt,'pi':np.pi,'e':np.e,'G':gauss,'Min':np.minimum, 'Max':np.maximum}]#,"mpmath"]


####### Importance Sampling from Markov chain ###############
def Ns(delta,epsilon,dF):
    return int(np.ceil(dF**2*log(2/delta)/(2*epsilon**2)))


def solveIneqSymb(phi,u,f=G):  # phi is a conjunction of linear inequalities of the form a1*u+b1<=(<,>,>=)a2*u+b2 
    if f==U:
        LL=0
        UL=1
    elif f==G:
        LL=float(-np.inf)
        UL=float(np.inf)
    else:
        print("Error: distribution not supported")
        return None
    if phi==True:
        return LL,UL
    if phi==False:
        return 0,0
    if phi.func==And:
        ineqs=list(phi.args)
    else:
        ineqs=[phi]     
    solphi=solve(ineqs,u)
    if solphi==False:
        return 0,0
    if solphi==True:
        return LL,UL
    if solphi.func==And:
        ineqs=list(solphi.args)
    else:
        ineqs=[solphi]    
    umin=[LL]
    umax=[UL]
    for ineq in ineqs:
        cmp,lhs,rhs=ineq.func,ineq.args[0],ineq.args[1]
        if cmp==StrictLessThan or cmp==LessThan:
            if rhs==u:
                umin.append(lhs)
            else:
                umax.append(rhs)
        else:
            if rhs==u:
                umax.append(lhs)
            else:
                umin.append(rhs)        
    return Max(*umin),Min(*umax)



def sequentialRejection_vect(e,drawList,varorder=None,delta=None,N=10,maxe=1):
    start_time=time.time()
    if varorder==None:
        varorder=[ ee.args[0] for ee in drawList]
    k=len(varorder)    
    printv(k," variables: ", varorder)
    varindex={x:i for x,i in zip(varorder,range(k))}   
    zz = np.zeros(N)
    varState=np.zeros((k,N),dtype=np.float64)   # state variable is a matrix k x N = n.variables x N
    logWeights=np.zeros((1,N),dtype=np.float64)  # sum of (log) weights, one for each sample
    for drawtr in drawList:
        u=drawtr.args[0]
        i=varindex[u]
        phi=drawtr.args[1]
        rho=drawtr.args[2]
        if phi==False:
            print("Warning, zero measure branch determined by condition ",phi)
            return 0, 0, 0, 0
        printv(phi)
        f=rho.func
        LL,UL=solveIneqSymb(phi,u,f)
        rho1=f(*(rho.args+(LL,UL)))            
        printv(rho1)
        broad_args=tuple(a+z for a in rho1.args) # summing a zero vector z to force broadcasting of constants in arguments
        fargs=lambdify(varorder+[z],broad_args,modules=mymathmodule)
        
        if f==U: 
            varState[i,:]=uniform(*fargs(*varState,zz))
            logWeights+=log_cdfU(*fargs(*varState,zz))
        elif f==G:
            varState[i,:]=trunc_norm(*fargs(*varState,zz))
            logWeights+=log_cdfG(*fargs(*varState,zz))
        else:
            print("Error: distribution not supported")
            return None         
        printv(varState)
        printv(logWeights)  
    w_den = np.exp(logWeights)
    v_den = w_den.sum()/N
    eva_den=((w_den-v_den)**2).sum()/N
    if e!=1 and e!=0:
        fe=lambdify(varorder,e)
        fev=fe(*varState)
        w_num=w_den*fev
        v_num=w_num.sum()/N
        eva_num=((w_num-v_num)**2).sum()/N
    elif e==1:
        v_num=v_den
        eva_num=eva_den
    else:
        v_num=0
        eva_num=0
    eveb_den=sqrt(2*eva_den*log(3/delta)/N)+3*log(3/delta)/N
    eveb_num=sqrt(2*eva_num*log(3/delta)/N)+3*maxe*log(3/delta)/N
    
    printv("Overall n. of samples for this branch:   ", N*k)
    printv("Empirical variance (num.):               ", eva_num)
    printv("Empirical variance (den.):               ", eva_den)
    printv("Empirical Bernstein error bound (num.):  ",eveb_num)
    printv("Empirical Bernstein error bound (den.):  ",eveb_den)
    
    return v_num, v_den, eveb_num, eveb_den


def semPP_3_r(P,e,delta=0.001,N=100,maxe=1,k=7,simp=False,optbound=False,epsilon=.01,truncL=0,truncR=1,baseIntv=[0,1]):
    start_time=time.time()
    global varcount
    varcount=0
    CS=collectS_s(e,[],True,P,k=k,simp=simp)
    print('N. of branches=  ',len(CS))
    num=0
    den=0
    eveblist=[]
    eveblist_den=[]
    nbranch=0
    termprob=0
    for e0,dL,term in CS:
        dL,vo=preprocessBranch(dL,simp=simp)
        if dL!=[]:
            print('Integrand: '+str(e0) +' *', [(e.args[2],e.args[1]) for e in dL], term*' (terminated)'+(not term)*' (not terminated)')
            N0=N
            Hoeff=False
            if optbound:
                wb=findBoundW_A(dL,vo,e0,truncL=truncL,truncR=truncR,baseIntv=baseIntv) # find bound on maximum weight
                if wb==0:
                    print('   *0 variance bound, setting N=1 for current branch*')                        
                    N0=1
                else:
                    N0=Ns(delta,epsilon,wb)
                if N0>N:
                    N0=N
                else:
                    Hoeff=True
                    print('   *N. of samples from Hoeffding with given epsilon and delta =*',N0)                        
            vnum, vden,  eveb_num, eveb_den = sequentialRejection_vect(e0,dL,varorder=vo,delta=delta,N=N0,maxe=maxe)#sequentialRejection_C(e0,dL,varorder=vo,N=Nnum,weightbranch=weightbranch,delta=delta,epsilon=epsNum,baseIntv=baseIntv,splitting=splitting,truncL=truncL,truncR=truncR)
            if vden>0:
                if N0==1:
                    pass
                elif Hoeff:
                    if term and e0!=0:
                        eveblist.append(epsilon)
                    eveblist_den.append(epsilon)
                    nbranch+=1
                else:
                    if term and e0!=0:
                        eveblist.append(eveb_num)
                    eveblist_den.append(eveb_den)
                    nbranch+=1
                if term:
                    num+=vnum
                    termprob+=vden
                den+=vden
                nbranch+=1
    
    termprob=termprob/den
    print('Numerator estimate:                ',num)
    print('Denumerator estimate:              ',den)
    print('Termination probability estimate:  ',termprob)
    print('Estimated expectation of '+str(e)+' (weighing 0 non-terminated branches):',num/den)
    epsNum=sum(eveblist)
    epsDen=sum(eveblist_den)
    ci=[(num-epsNum)/(den+epsDen),(num+epsNum)/(den-epsDen)]
    print('Confidence Interval:               ',ci)
    print('Error probability (delta) bound:   ',delta*nbranch)
    print('N. on nonzero branches (num+den):  ',nbranch)
    print("TOTAL time       %s seconds -------" % (time.time()-start_time))
    return num/den,ci,delta*nbranch


def bern(x,e=1/2,vL=[0,1]):   # Bernoulli with success prob. e (expression)
    S=seq(drawv(x,U()),ifte(x>=e,setv(x,vL[0]),setv(x,vL[1])))
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
semPP_3_r(BA,burglary,delta=0.001/8,N=1000000,k=20,maxe=1)




#------------------ Example 2: TrueSkill model


#model definition
sk=var('skillA, skillB, skillC perfA1,perfB1,perfB2, perfC2,perfA3,perfC3')
uv=var('u1 u2 u3 u4 u5 u6 u7 u8 u9')
varcount=0
scale=100 # scale factor, can be adjusted
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
semPP_3_r(TrueSkill,skillA,delta=0.001/2,N=5000,k=10,maxe=1.3)

