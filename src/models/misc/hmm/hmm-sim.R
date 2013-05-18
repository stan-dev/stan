library("rstan")
source("hmm.data.R")
fit <- stan('hmm-semisup.stan', # 'hmm-fit-semisup.stan',
            data=list(K=K,V=V,T=T,T_unsup=T_unsup,w=w,z=z,u=u,alpha=alpha,beta=beta),
            iter=200, chains=1, init=0);  # fit = fit // reuse model
