      double precision function poster(nparm,parm)
c user-supplied function to evaluate logposterior      
c first time: reads data and forms crude parameter estimates
      implicit integer (a-z)
      parameter(mxparm=200,mxn=500)
      double precision parm(mxparm), logpost
      double precision y1(mxn), y2(mxn), ddet(5), ldet(5), sigmat(5,3),
     1     delta1, delta2, phi, dev1, dev2, sig, sigphi, sigsquig
      integer inc1(mxn), inc2(mxn), incs(mxn)
      character cyear*4, file*13
      data ncalls, n /0, 0/

      ncalls = ncalls+1
      if (nparm.eq.-1) then
c     first time: read any data and starting values
c        read (5,*) cyear
c        file = "metrop" // cyear // ".inp"
         open (10, file="metrop1920.inp")
         read (10,*) year, n
         read (10,*) (y1(i), y2(i), inc1(i), inc2(i), i=1,n)
         do i=1,n
            incs(i) = 1 + abs(inc1(i)) + 2*abs(inc2(i))
            if (inc1(i)*inc2(i).eq.-1) incs(i) = 5
         end do
c     starting values
         parm(1) = .5
         parm(2) = .5
         parm(3) = .05
         parm(4) = log(.06)
         parm(5) = log(parm(3))
         parm(6) = log(.1)
         nparm = 6
      end if

      delta1 = parm(1)
      delta2 = parm(2)
      phi = parm(3)
      sig = dexp(2.*parm(4))
      sigphi = dexp(2.*parm(5))
      sigsquig = dexp(2.*parm(6))
c     1:  open, open
c     2:  inc, open
c     3:  open, inc
c     4:  inc, inc
c     5:  inc, inc (change of party; can only occur with special elections)
      do j=1,5
         sigmat(j,1) = sig + sigsquig
         sigmat(j,2) = sigsquig
         sigmat(j,3) = sig + sigsquig
      end do
      sigmat(2,1) = sigmat(2,1) + sigphi
      sigmat(4,1) = sigmat(4,1) + sigphi
      sigmat(5,1) = sigmat(5,1) + sigphi

      sigmat(3,3) = sigmat(3,3) + sigphi
      sigmat(4,3) = sigmat(4,3) + sigphi
      sigmat(5,3) = sigmat(5,3) + sigphi

      sigmat(4,2) = sigmat(4,2) + sigphi

      do j=1,5
         ddet(j) = sigmat(j,1)*sigmat(j,3) - sigmat(j,2)**2.
         ldet(j) = dlog(ddet(j))
      end do
      logpost = .5*(dlog(sigphi) + dlog(sigsquig))
      do i=1,n
         ii = incs(i)
         dev1 = y1(i) - delta1 - phi*inc1(i)
         dev2 = y2(i) - delta2 - phi*inc2(i)
         logpost = logpost -.5*ldet(ii) - .5*(sigmat(ii,3)*dev1**2. +
     1      sigmat(ii,1)*dev2**2. - 2*sigmat(ii,2)*dev1*dev2)/ddet(ii)
      end do
      write (6,200) ncalls, (parm(i),i=1,6), logpost
 200  format (i5, 6f9.3, f10.2)
      poster = logpost
      return
      end





