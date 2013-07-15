c general purpose Bayesian calculation program
c
c input: poster = fortran function (at the end of this program) to 
c                 compute posterior distribution at a given set of parameter
c                 values.  First call is to read data and create crude 
c                 parameter estimates
c        data file : first line contains: nseq=# of parallel metrop sequences
c                                         ndraws=length of sequences
c                                         seed = random seed (0 to 2147483647)
c                    subsequent lines contain data to be read by poster
c  
c output: crude parameter estimates
c         posterior mode (and approx s.e. from 2nd derivative matrix)
c         metropolis acceptance rate
c         metropolis/Gelman-Rubin style output
c
c parameter statement sets maximum allowed:
c    number of parameters 
c    number of Metrop sequences
c    number of draws per sequence (must be even #)
c    number of values for final inferences = mxseq*mxdraw/2
      implicit integer (a-z)
      parameter(mxparm=50,mxseq=20,mxdraw=100000,mxtot=1000000)
      double precision parm(mxparm),secder(mxparm,mxparm)
      double precision invsecdr(mxparm,mxparm),covar(mxparm,mxparm)
      double precision poster,lh,draws(mxparm+1,mxdraw,mxseq)
      double precision summ(mxparm+1,10),crudsumm(mxparm+1,3)
      double precision start(mxseq,mxparm),xp1,xp2,xv
c read main control parameters and write first line of output
      read (5,*) nseq,ndraws,seed,alt,xp1,xp2,xv
c read nseq, ndraws, seed, alt, xv=variance for fixed parameters
      read (4,*) nseq,ndraws,seed,alt,xv
      if (nseq.gt.mxseq.or.ndraws.gt.mxdraw) then
        write (6,189) 
189     format('too many sequences (>10) or draws (>2000) specified')
        go to 201
        end if        
      nd = ndraws - (ndraws/2)
      write (6,191) nseq,ndraws,seed,nd
191   format('METROPOLIS PROGRAM',5x,i2,' sequences of length ',i6,
     .       ' with random seed ',i10,/,'metropolis results use last ',
     .       i6,' values in each sequence')
      write (6,187) xp1,xp2,xv
187   format('mode log(var2): ',f4.1,2x,' mode z(r): ',f4.1,2x
     .       ' var of these: ',f4.2)
      write (6,192)
192   format(i10)
c generate crude estimates and evaluate likelihood at crude estimates
      nparm = -1
      lh = poster(nparm,parm)
      if (nparm.gt.mxparm) then
        write (6,188)
188     format('posterior subroutine used too many parameters (>200)')
        go to 201
        end if
c save crude estimates for output
      do 10 i = 1,nparm
        crudsumm(i,1) = parm(i)
 10     continue
      crudsumm(nparm+1,1) = lh
c Newton-Raphson w/ steepest ascent to reach mode
c    or alternatively enter mode and covariance matrix for t-approx
      if (alt.eq.0) call getmode(nparm,parm,secder,lh,err)
      if (alt.eq.1) then 
	read (5,*) (parm(i),i=1,47)
	parm(48) = xp1
	parm(49) = xp2
	lh = poster(nparm,parm)
	read (5,*) ((secder(i,j),j=1,47),i=1,47)
	do 1605 i = 1,47     
	  do 1605 j = 48,49
	    secder(i,j) = 0.d0
	    secder(j,i) = 0.d0
1605        continue
        secder(48,48) = 1.d0 / xv
        secder(48,49) = 0.d0
        secder(49,48) = 0.d0
        secder(49,49) = 1.d0 / xv
	go to 109
	end if 
      if (err.eq.1) then
        write (6,101) 
101     format ('program stopping -- failed to find mode')
        go to 201
        end if
c at mode: find second deriv matrix, then cholesky decomp of secder matrix,
c          (we find upper triangle R with R^T R = second derivative matrix)
c          then invert cholesky triangle, and finally compute covar matrix  
c          (we find upper triangle R with R^T R = second derivative matrix)
c          mode (inverse chol x inverse chol^T)
      call seconder(nparm,parm,secder)
109   call cholesky(secder,nparm,err)
      if (err.eq.1) then 
        write (6,102)
102     format ('program stopping -- 2nd deriv matrix is not < 0')
        go to 201
        end if
      call invuptri(secder,nparm,invsecdr)
      call getcovar(invsecdr,nparm,covar)
c collect post mode and approx s.e. for output  
      do 110 i = 1,nparm
        crudsumm(i,2) = parm(i)
        crudsumm(i,3) = dsqrt(covar(i,i))
110     continue
      crudsumm(nparm+1,2) = lh
c 
c METROPOLIS SECTION OF PROGRAM
c includes routines for getting starting values and convergence check 
c
c get starting values for the metropolis sequences 
      call cholesky(covar,nparm,err)
      call getstart(start,seed,nparm,parm,secder,covar,nseq)
c carry out metropolis -- check that acceptance rate is in correct range
      call metrop(nparm,parm,covar,nseq,start,ndraws,draws,err,seed)
      if (err.eq.2) write (6,104)
104   format ('never got accept rate in good range in 4 tries')
c carry out convergence check on the parallel metropolis sequences
      call convchk(draws,nseq,ndraws,nparm,summ,err)
      if (err.eq.3) write (6,103)
103   format ('metrop did not converge (R>1.1 for a param) -- rerun',
     .        ' with more draws')
c 
c write output
      write (6,197)
197   format(9x,' crud estm',' post mode',' asympt se',10x,
     .          ' metrop t-approxim: mean+95%ci')
      do 190 i = 1,nparm
        write (6,198) i,(crudsumm(i,j),j=1,3),(summ(i,j),j=3,5)
198     format ('parm',i3,2x,3(1x,f9.5),10x,3(1x,f9.5))
190     continue
      write (6,195) (crudsumm(nparm+1,j),j=1,2),(summ(nparm+1,j),j=3,5)
195   format ('likhood',2x,2(1x,e11.5),20x,3(1x,e11.5))
      write (6,192)
      write (6,196)
196   format(9x,' estim. R ','97.5%ile-R','  2.5%ile ','   25%ile ',
     .          '   50%ile ','   75%ile ',' 97.5%ile ')
      do 200 i = 1,nparm
        write (6,199) i,(summ(i,j),j=1,2),(summ(i,j),j=6,10)
199     format ('parm',i3,2x,7(1x,f9.5)) 
200     continue
      write (6,194) (summ(nparm+1,j),j=1,2),(summ(nparm+1,j),j=6,10)
194   format ('likhood',2x,7(1x,e11.5)) 
      write (6,192)
c     do 300 i = 1,nseq
c     do 300 j = 1,ndraws
cwrite (6,292) (draws(k,j,i),k=1,nparm+1)
c92     format(3f14.8,1x,e14.8)
c00     continue
201   stop 
      end

      subroutine metrop(nparm,parm,covar,nseq,start,ndraws,draws,
     .                  err,seed)
c generate nseq metropolis chains each of length ndraws
      implicit integer(a-z)
      parameter(mxparm=50,mxseq=20,mxdraw=100000)
      double precision parm(mxparm),covar(mxparm,mxparm)
      double precision poster,draws(mxparm+1,mxdraw,mxseq)
      double precision start(mxseq,mxparm),restart(mxseq,mxparm)
      double precision curr(mxparm),cand(mxparm)
      double precision curpst,newpst,ratio,chkpost(mxseq)
      double precision count(mxseq),scale,totchk,unif,rangen
      err = 0
c Gelman,Gilks,James result suggests this scale for normal
      scale = 2.4d0/dsqrt(dble(nparm))
c monitor metropolis acceptance rate after ncheck draws (unless ndraws is small)
      ncheck = 100
      if (ndraws.le.2*ncheck) ncheck = ndraws
c
c metropolis loop -- 1 until time to check on accept rate
      fail = 0
11    totchk = 0.d0
      do 200 i = 1,nseq
        count(i) = 0
        do 1 ii = 1,nparm
          curr(ii) = start(i,ii)
 1        continue
        curpst = poster(nparm,curr)
        do 100 j = 1,ncheck
c generate candidate from mvn with chosen scale
          call getcand(nparm,curr,covar,cand,scale,seed)
          newpst = poster(nparm,cand)
          ratio = dexp(newpst-curpst)
c accept/reject candidate
          unif = 0.d0
          if (ratio.lt.1.d0) unif=rangen(seed)
          if (unif.lt.ratio) then
            curpst = newpst
            count(i) = count(i) + 1
            do 10 jj = 1,nparm
              curr(jj) = cand(jj)
 10           continue
            end if
c save draw
            do 20 jj = 1,nparm
              draws(jj,j,i) = curr(jj)
 20           continue
            draws(nparm+1,j,i) = curpst
100       continue
c if accept rate check is being performed save last draw and posterior 
        if (ncheck.lt.ndraws) then
          chkpost(i) = curpst
          do 30 jj = 1,nparm
            restart(i,jj) = curr(jj)
 30         continue
          totchk = totchk + count(i)
          end if
200     continue
c compute, evaluate, and write metropolis acceptance rate
      if (ncheck.lt.ndraws) then             
        totchk = totchk / (nseq*ncheck)
        write (6,250) totchk,ncheck
250     format('metrop accept rate is ',f6.3,' after ',i4,' draws/seq')
        if (totchk.lt.0.1d0.or.totchk.gt.0.5d0) then
          fail = fail + 1
          if (fail.eq.4)  then
            err = 2 
            end if
          if (totchk.gt.0.5d0) scale = scale * 1.25d0
          if (totchk.lt.0.1d0) scale = scale / 1.25d0
          go to 11
          end if
c resume metropolis after check (whether accept rate is OK or not)
251     do 290 i = 1,nseq
          curpst = chkpost(i)
          do 261 ii = 1,nparm
            curr(ii) = restart(i,ii)
261         continue
          do 280 j = ncheck+1,ndraws
            call getcand(nparm,curr,covar,cand,scale,seed)
            newpst = poster(nparm,cand)
            ratio = dexp(newpst-curpst)
            unif = 0.d0
            if (ratio.lt.1.d0) unif=rangen(seed)
            if (unif.lt.ratio) then
              curpst = newpst
              count(i) = count(i) + 1
              do 275 jj = 1,nparm
                curr(jj) = cand(jj)
275             continue
              end if
              do 279 jj = 1,nparm
                draws(jj,j,i) = curr(jj)
279             continue
              draws(nparm+1,j,i) = curpst
280         continue
          write (6,285) i
285       format ('done with sequence ',i2)
290       continue
        end if
291   totchk = 0.d0
      do 292 i = 1,nseq
        totchk = totchk + count(i)
292     continue
      totchk = totchk / (nseq*ndraws)
      write (6,250) totchk,ndraws
      return
      end

      subroutine convchk(draws,nseq,ndraws,nparm,summ,err)
c assess convergence of metropolis draws
      implicit integer(a-z)
      parameter(mxparm=50,mxseq=20,mxdraw=100000,mxtot=1000000)
      double precision draws(mxparm+1,mxdraw,mxseq)
      double precision summ(mxparm+1,10),sum,sum2,x,mn,s2,vec(mxtot)
      double precision summn,sums2,summnsq,sums2sq,desc(5)
      double precision summns2,summn2s2,temp,fq,qf,tq,qt,df1
      double precision W,muhat,B,varW,varB,covWB,postvar
      double precision sig2hat,varpostvar,postdf,varlodf
      ni = ndraws/2
      nd = ndraws - ni
c for each scaler parameter
      do 1000 ip = 1,nparm+1
        count = 0
        summn = 0.d0
        sums2 = 0.d0
        summnsq = 0.d0
        sums2sq = 0.d0
        summns2 = 0.d0 
        summn2s2 = 0.d0 
c accumulate mean and s.d within each sequence
      do 100 i = 1,nseq
        sum = 0.d0
        sum2 = 0.d0
        do 90 j = 1,nd
          count = count + 1 
          x = draws(ip,ni+j,i)
          vec(count) = x
          sum = sum + x
          sum2 = sum2 + x*x
 90       continue
        mn = sum /nd
        s2 = (sum2 - sum*mn) / (nd-1)
c accumulate sums, cross products needed for Gelman-Rubin scale reduction factor
        summn = summn + mn
        summnsq = summnsq + mn*mn
        sums2 = sums2 + s2
        sums2sq = sums2sq + s2*s2
        summns2 = summns2 + mn*s2
        summn2s2 = summn2s2 + mn*mn*s2
100     continue
c G-R calculations
      W = sums2 / nseq
      muhat = summn / nseq
      B = nd*(summnsq - summn*muhat)/(nseq-1)
      varW = (sums2sq - sums2*W)/((nseq-1)*nseq)
      varB = 2.d0*B*B/(nseq-1)
      covWB = nd*( ((summn2s2 - summnsq*W)/(nseq-1)) -
     .       ((summns2 - summn*W)*2.d0*muhat/(nseq-1)) )/nseq
      sig2hat = ((nd-1.d0)*W + B)/nd
      postvar = sig2hat + B/(nd*nseq)
      varpostvar = (((nd-1)**2)*varW + ((nseq + 1)**2)*varB/(nseq*nseq)
     .               + 2*(nd-1)*(nseq+1)*covWB/nseq)/(nd*nd)
      postdf = 2*postvar*postvar / varpostvar    
      tq = qt(postdf)
c compute conservative t approximation
      summ(ip,3) = muhat - tq*sqrt(postvar)
      summ(ip,4) = muhat
      summ(ip,5) = muhat + tq*sqrt(postvar)
c compute quantiles 
      call analy(vec,count,desc)
      do 987 jj = 1,5 
        summ(ip,jj+5) = desc(jj)
987     continue
c compute scale reduction factor and .975 point
c currently uses a fairly lame F-distn quantile approx
      varlodf = 2*W*W /varW
      df1 = nseq - 1.d0
      fq = qf(df1,varlodf)
      summ(ip,1) = dsqrt(postvar*postdf/(W*(postdf-2.d0)))
      temp = (nseq + 1.d0)*(B/W)/(nseq*nd)
      temp = ((nd-1.d0)/nd) + temp*fq 
      summ(ip,2) = dsqrt( temp*postdf/(postdf-2.d0) )
      if (summ(ip,1).gt.1.1d0) err = 3
1000  continue
      return
      end
 
      subroutine getmode(nparm,parm,secder,lh,merr)
c find mode of posterior distribution
      parameter(mxparm=50)
      implicit integer(a-z)
      double precision parm(mxparm),secder(mxparm,mxparm)
      double precision tmpparm(mxparm),parminc(mxparm)
      double precision der1(mxparm),poster
      double precision toler,ztol,lh,oldlh
      double precision sum,t
      data toler/1.d-12/,ztol/1.d-20/
      data mxloop/100/
      merr = 0
      end = 0
      lh = -999999.d99
      do 10 loop=1,mxloop
        oldlh = lh
c  compute vector of first derivatives
        call firstder(nparm,parm,der1)
c  compute negative of second derivative matrix numerically
        call seconder(nparm,parm,secder)
c  Newton-Raphson step
c  --get cholesky decomposition of secder (writing over secder)
        call cholesky(secder,nparm,err)
        if (err.eq.0) then
c  --solve for Newton step unless Cholesky decomp fails 
          do 50 i=1,nparm
            sum = 0.d0
            do 49 kk = 1,i-1
              sum = sum + secder(kk,i)*parminc(kk)
 49           continue
            parminc(i) = (der1(i) - sum) / secder(i,i)
 50         continue
          do 60 i=1,nparm
            ii = nparm + 1 - i
            sum = 0.d0
            do 59 kk = ii+1,nparm
              sum = sum + secder(ii,kk)*parminc(kk)
 59           continue
            parminc(ii) = (parminc(ii) - sum) / secder(ii,ii)
 60         continue
          t = 1.d0
        else            
c can not do Newton-Raphson step -- use steepest ascent step
          do 70 i=1,nparm
            parminc(i) = der1(i)
  70        continue
          t = 0.1d0
        end if
c check that step increases posterior density or reduce step (mainly for  
c                                                             steep ascent)
2003    do 2004 i = 1,nparm
2004      tmpparm(i) = parm(i) + parminc(i)*t
          lh = poster(nparm,tmpparm)
          if (lh.le.oldlh) then
          t = t / 2.d0
          if (t.lt.ztol) then
            t = 0.d0
            go to 2009
            end if
          go to 2003
          end if
c found a good step
2009    do 2010 i = 1,nparm
2010      parm(i) = parm(i) + t*parminc(i)
c test for convergence
        if (lh.lt.(oldlh+toler)) then
          end = 1
          go to 101
          end if
 10     continue
      merr = 1
      write (6,102) mxloop,oldlh,lh
 102  format(' did not find max in ',i3,' steps',/,
     .       'poster dens for last two steps:',2e15.8)
 101  return    
      end 

      subroutine getstart(start,seed,nparm,parm,secder,covar,nseq)  
c get starting values for metropolis by impt wtd sampling from t_4 approx
      parameter(mxparm=50,mxseq=20,mximpt=2000)
      implicit integer(a-z)
      double precision parm(mxparm),secder(mxparm,mxparm)
      double precision covar(mxparm,mxparm)
      double precision gaudev,gamdev,sum,bigset(mximpt,mxparm)
      double precision norms(mxparm),poster,new(mxparm),tdens
      double precision rangen,uni,start(mxseq,mxparm)
      double precision fac,sumwt,thewt(mximpt),bigwt(mximpt)
      double precision aa,bb
      data four/4/,two/2/
c generate 1000 draws from t_4 approx at post mode
      sumwt = 0.d0
      do 1000 i = 1,mximpt
        do 10 ii = 1,nparm
          norms(ii) = gaudev(seed)
 10       continue
        fac = dsqrt(2.d0/gamdev(two,seed))
        do 100 ii = 1,nparm
          sum = 0.d0
          do 99 j = 1,ii      
            sum = sum + covar(j,ii)*norms(j)
 99         continue
          new(ii) = parm(ii) + sum*fac
          bigset(i,ii) = new(ii)
100       continue
c compute importance ratio
        thewt(i) = dexp(poster(nparm,new)+5900- 
     .                  tdens(new,parm,secder,nparm,four))
        sumwt = sumwt + thewt(i)
        bigwt(i) = sumwt
1000    continue 
c sample nseq starting values using importance ratios as weights (w/out replac)
      do 2000 i = 1,nseq
        uni = rangen(seed)*sumwt
c binary search to locate sampled value
        call locate(bigwt,mximpt,uni,ix)
        do 1100 j = 1,nparm
          start(i,j) = bigset(ix,j)
1100      continue
        sumwt = sumwt - thewt(ix)
        do 1200 k = ix,mximpt
          bigwt(k) = bigwt(k) - thewt(ix)
1200      continue
2000    continue
      return
      end

      subroutine getcand(nparm,curr,vcov,cand,scale,seed)
c generate candidate value for metropolis -- using normal jumping distn with
c                                            s.d. multiplied by scale
      parameter(mxparm=50)
      implicit integer(a-z)
      double precision curr(mxparm),cand(mxparm),vcov(mxparm,mxparm)
      double precision scale,gaudev,norms(mxparm),sum
      do 10 i = 1,nparm
        norms(i) = gaudev(seed)
 10     continue
      do 100 i = 1,nparm
        sum = 0.d0
        do 99 j = 1,i      
          sum = sum + vcov(j,i)*norms(j)
 99       continue
        cand(i) = curr(i) + scale*sum
100     continue
      return
      end

      double precision function tdens(x,mu,sigma,dim,df)
c evaluate the t-density used to generate Metropolis starts
      implicit integer (a-z) 
      parameter(mxparm=50)
      double precision x(mxparm),mu(mxparm),sigma(mxparm,mxparm)
      double precision sum,sumvsq,rdf
      rdf = df
      sumvsq = 0.d0
      do 60 i = 1,dim
        sum = 0.d0 
        do 59 kk = i,dim
          sum = sum + sigma(i,kk)*(x(kk)-mu(kk))
 59       continue
        sumvsq = sumvsq + sum*sum
 60     continue
      tdens = -0.5d0*(dim+df)*dlog(1.d0 + (sumvsq/rdf))
      return
      end
      
      subroutine firstder(nparm,parm,der1)
c numerically evaluate vector of first derivatives of posterior distn
      implicit integer (a-z)
      parameter(mxparm=50)
      double precision step,delta,parm(mxparm),der1(mxparm)
      double precision l1,l2,poster
      data step/1.d-04/
      do 10 i=1,nparm
        delta = step*parm(i)
        parm(i) = parm(i) + delta
        l1 = poster(nparm,parm)
        parm(i) = parm(i) - 2.d0*delta
        l2 = poster(nparm,parm)
        parm(i) = parm(i) + delta
        der1(i) = (l1 - l2)/(2.d0*delta)
 10     continue
      return
      end

      subroutine seconder(nparm,parm,secder)
c numerically evaluate matrix of second derivatives of posterior distn
c (actually finds negative of second derivs --> inverse of asymp covar matrix)
      implicit integer (a-z)
      parameter(mxparm=50)
      double precision parm(mxparm),secder(mxparm,mxparm)
      double precision step,delta1,delta2,poster
      double precision l1,l2,l3,l4
      data step/1.d-04/
      do 10 i = 1,nparm
        delta1 = step*parm(i)
        do 11 j = i,nparm
          delta2 = step*parm(j)
          parm(i) = parm(i) + delta1
          parm(j) = parm(j) + delta2
          l1 = poster(nparm,parm)
          parm(j) = parm(j) - 2.d0*delta2
          l2 = poster(nparm,parm)
          parm(i) = parm(i) - 2.d0*delta1
          l3 = poster(nparm,parm)
          parm(j) = parm(j) + 2.d0*delta2
          l4 = poster(nparm,parm)
          secder(i,j) = -1.d0*(l1 + l3 - l2 - l4)/(4.d0*delta1*delta2)
          parm(i) = parm(i) + delta1
          parm(j) = parm(j) - delta2
 11       continue
        secder(j,i) = secder(i,j)
 10     continue
      return
      end

      subroutine getcovar(invchlpr,n,covar)
c compute asymptotic covar matrix of normal/t approx 
c --- if R is cholesky decomp of second deriv matrix and T = R^(-1)
c --- then this finds T T^t 
      implicit integer (a-z)
      parameter(mxparm=50)
      double precision invchlpr(mxparm,mxparm),covar(mxparm,mxparm)
      double precision sum
      do 200 i = 1,n
        do 199 j = i,n
          sum = 0.d0 
          do 198 k = j,n
            sum = sum + invchlpr(i,k)*invchlpr(j,k)
 198        continue
          covar(i,j) = sum
          if (j.ne.i) covar(j,i) = sum
 199      continue
 200    continue
      return
      end

      subroutine invuptri(x,n,y)
c compute inverse of upper triangular matrix (used here to invert cholesky 
c                                             triangle of second deriv matrix)
      implicit integer (a-z)
      parameter(mxparm=50)
      double precision x(mxparm,mxparm),y(mxparm,mxparm),sum 
      do 200 i = 1,n
        j = n + 1 - i
        y(j,j) = 1.d0 / x(j,j)
        do 200 ii=1,j-1
          k = j - ii
          sum = 0.d0
          do 199 iii = k+1,j
            sum = sum + x(k,iii)*y(iii,j)
 199        continue
          y(k,j) = -1.d0*sum/x(k,k)
 200      continue
      return
      end

      subroutine cholesky(x,n,err)
c compute cholesky decomposition of the matrix x
      implicit integer(a-z)
      parameter(mxparm=50)
      double precision x(mxparm,mxparm),sum
      err = 0
      do 32 i = 1,n
        sum = 0.d0
        do 29 kk = 1,i-1
          sum = sum + x(kk,i)*x(kk,i)
 29       continue
        if (x(i,i).le.sum) then
          err = 1
          return
          end if
        x(i,i) =dsqrt(x(i,i)-sum)
        do 31 j = i+1,n
          sum = 0.d0
          do 30 kk = 1,i-1
            sum = sum + x(kk,j)*x(kk,i)
 30         continue
          x(i,j) = (x(i,j) - sum)/x(i,i)
 31       continue
 32     continue
      do 40 i = 2,n
        do 40 j = 1,i-1
          x(i,j) = 0.d0
40        continue
      return
      end

      subroutine locate(xx,n,x,j)
c binary search routine (used to carry out importance weighted samples)
      parameter(mximpt=2000)
      implicit integer (a-z)
      double precision xx(mximpt),x
      jl = 0
      ju = n
  10  if ((ju-jl).gt.1.) then
        jm = (ju + jl)/2
        if (x.gt.xx(jm)) then
          jl = jm
        else 
          ju = jm
        end if
      go to 10
      end if
      j = ju
      return
      end

      subroutine analy(x,n,desc)
c sorts vector and obtains quantiles (2.5, 25, 50, 75, 97.5)       
      parameter(mxtot=1000000)
      implicit integer(a-z)
      double precision x(mxtot)
      double precision desc(5),ixr,r
      call hpsort(n,x)
      ix = (n+1)/40 
      ixr = dble(n+1)/40 
      r = ixr - ix
      desc(1) = (1.0-r)*x(ix) + r*x(ix+1)
      ix = (n+1)/4  
      ixr = dble(n+1)/4  
      r = ixr - ix
      desc(2) = (1.0-r)*x(ix) + r*x(ix+1)
      ix = (n+1)/2    
      ixr = dble(n+1)/2   
      r = ixr - ix
      desc(3) = (1.0-r)*x(ix) + r*x(ix+1)
      ix = 3*(n+1)/4     
      ixr = 3*dble(n+1)/4
      r = ixr - ix
      desc(4) = (1.0-r)*x(ix) + r*x(ix+1)
      ix = 39*(n+1)/40  
      ixr = 39*dble(n+1)/40   
      r = ixr - ix
      desc(5) = (1.0-r)*x(ix) + r*x(ix+1)
      return
      end 

      subroutine hpsort(n,ra)
c heap sort from Numerical recipes
      implicit integer (a-z)
      parameter(mxtot=1000000)
      double precision rra,ra(mxtot)
      l = (n/2) + 1
      ir = n
  10  continue
        if (l.gt.1) then
          l = l - 1
          rra=ra(l)
        else
          rra=ra(ir)
          ra(ir)=ra(1)
          ir = ir - 1
          if (ir.eq.1) then
            ra(1) = rra
            return
          end if
        end if
        i = l
        j = l + l
 20     if (j.le.ir) then
          if (j.lt.ir) then
            if (ra(j).lt.ra(j+1)) j = j + 1
          end if
          if (rra.lt.ra(j)) then
            ra(i)=ra(j)
            i = j
            j = j + j
          else
            j = ir + 1
          end if
          go to 20
        end if
        ra(i) = rra
      go to 10
      end

      double precision function rangen(ix)
c portable uniform random number generator -- mult=16807, base=2**31
      double precision m
      integer a,p,ix,b15,b16,xhi,xalo,leftlo,fhi,k
      data a/16807/,b15/32768/,b16/65536/,p/2147483647/
      data m/4.6566128752458e-10/
      xhi = ix/b16
      xalo = (ix-xhi*b16)*a
      leftlo = xalo/b16
      fhi = xhi*a + leftlo
      k = fhi / b15
      oldix = ix
      ix = (((xalo - leftlo*b16) - p) + (fhi-k*b15)*b16) + k
      if (ix.lt.0) ix = ix + p
      rangen = dble(ix)*m
      if (rangen.ge.1.d0.or.rangen.le.0.d0) write (6,1)
1       format('apparent problem in rangen')
      return
      end

      double precision function gamdev(alph,iseed)
c generate gamma random variables (shape parameter = alph is an integer)
c from Numerical Recipes 
      implicit integer (a-z)
      double precision x,rangen,y,s,e,pi,alphm1
      data pi/3.141592654d0/
      if (alph.le.0) then
        write (6,1)
  1     format ('problem in gamdev')
        x = 0.d0
      else if (alph.lt.6) then
        x = 1.d0
        do 11 j = 1,alph
          x = x*rangen(iseed)
 11       continue
        x = -dlog(x)
      else
 21     y = tan(pi*rangen(iseed))
        alphm1 = alph - 1.d0
        s = dsqrt(2.d0*alphm1+1.d0)
        x = s*y + alphm1
        if (x.le.0.d0) go to 21
        e = (1.d0+y*y)*dexp(alphm1*dlog(x/alphm1)-s*y)
        if (rangen(iseed).gt.e) go to 21
      end if
      gamdev = x
      return
      end

      double precision function gaudev(ix)
c generate normal random variable
      implicit integer (a-z)
      double precision v1,v2,r,gset,fac,rangen
      data iset/0/
      if (iset.eq.0) then
 1      v1 = 2.d0*rangen(ix) - 1.d0
        v2 = 2.d0*rangen(ix) - 1.d0
        r = v1**2 + v2**2
        if (r.ge.1.d0) go to 1
        fac = dsqrt(-2.d0*dlog(r)/r)
        gset = v1*fac
        gaudev= v2*fac
        iset = 1
      else
        gaudev = gset
        iset = 0
      endif
      return
      end

      double precision function qt(df)
c from Abramowitz and Stegun -- inverse function for t-distn 
c currently hardwired for .975 pct point -- to change just replace xp
      implicit integer (a-z)
      double precision df,xp,xp3,xp5,xp7,xp9
      data xp/1.959964d0/
      xp3 = xp*xp*xp
      xp5 = xp3*xp*xp
      xp7 = xp5*xp*xp
      xp9 = xp7*xp*xp
      qt = xp + (xp3+xp)/(4.d0*df) + 
     .          (5*xp5 + 16*xp3 + 3*xp)/(96.d0*df*df) +
     .          (3*xp7 + 19*xp5 + 17*xp3  - 15*xp)/(384.d0*df*df*df) +
     .          (79*xp9 + 776*xp7 + 1482*xp5 - 1920*xp3 - 945*xp)/
     .                                         (92160.d0*df*df*df*df)
      return
      end
 
      double precision function qf(df1,df2)
c from Abramowitz and Stegun -- approx to inverse function for F-distn 
c approximation doesn't look too good
c currently hardwired for .975 pct point -- to change just replace xp
      double precision df1,df2,a,b,lam,h,w,xp
      data xp/1.959964d0/
      a = df2 / 2.d0
      b = df1 / 2.d0
      lam = (xp*xp - 3.d0) / 6.d0
      h = 2.d0 / ( (1.d0/(df1-1.d0)) + (1.d0/(df2-1.d0)) )
      w = xp*sqrt(h + lam)/h - (lam+(5.d0/6.d0)-(2.d0/(3.d0*h)))*
     .                         ((1.d0/(df1-1.d0))-(1.d0/(df2-1.d0)))
      qf = dexp(2.d0*w)
      return
      end

      double precision function invlogt(x)
      implicit integer (a-z)
      double precision x,t
      t= dexp(x)
      invlogt = t/(1.d0+t)
      return 
      end

      double precision function logit(y,n)
      implicit integer (a-z)
      double precision x,yy
      yy = y
      if (y.eq.0) yy = 0.5d0
      if (y.eq.n) yy = n - 0.5d0
      x = yy/n                   
      logit = dlog(x/(1.d0-x))
      return 
      end
