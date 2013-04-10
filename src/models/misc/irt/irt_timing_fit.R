library('rstan');
NMax <- 1000000;
fit <- 0;
Nchains <- 1;
Niter <- 200;
Js <- c(10,100,1000,10000);
Ks <- c(10,100,1000);
times <- array(NA,c(length(Js),length(Ks)));
for (jidx in 1:length(Js)) {
  J <- Js[jidx];
  for (kidx in 1:length(Ks)) {
    K <- Ks[kidx];
    if (Js[jidx] * Ks[kidx] <= NMax) {
      source('irt_sim.R');
      if (jidx == 1 && kidx == 1) {
        fit <- stan(file='irt.stan',
                    data=list(J=J,K=K,N=N,jj=jj,kk=kk,y=y),
                    init=0, seed = 23, 
                    iter=2,
                    chains=1)
      }
      t_start <- proc.time()[3];
      print(paste("J=",J,", K=",K," N=",J*K,sep=""),quote=F);
      fit <- stan(file='irt.stan',
                  data=list(J=J,K=K,N=N,jj=jj,kk=kk,y=y),
                fit=fit,
                # init=0,
                iter=Niter, 
                chains=Nchains,
                seed=23);
      t_end <- proc.time()[3];
      t_elapsed <- t_end - t_start;
      times[jidx,kidx] <- t_elapsed / Nchains / (Niter/2);
    }
  }
}

print("raters,items,N,sec/iter",quote=F);
for (jidx in 1:length(Js)) {
  for (kidx in 1:length(Ks)) {
    if (Js[jidx] * Ks[kidx] <= NMax) {
      print(paste(Js[jidx],Ks[kidx],Js[jidx]*Ks[kidx],times[jidx,kidx],        
                  sep=","), 
            quote=F);
    }
  }
}
