/* Stan model to compute 2-cmt PK model with IV and infusions
 *
 * Author: S. Weber
 * Date:   27th February 2015
 *
 * Input data is expected to be nonmem formatted. Parameters are
 * ALWAYS fit in LOG-space. Default parametrization is based on rate
 * constants as:
 *
 * ltheta[1] = k
 * ltheta[2] = k21
 * ltheta[3] = (k + k12 + k21)/3
 * ltheta[4] = V1
 *
 * Alternative parametrization is
 *
 * ltheta[1] = CL
 * ltheta[2] = V1
 * ltheta[3] = Q
 * ltheta[4] = V2
 *
 * Please refer to the evaluate_model function and choose between
 * trans (rate constants) and Qtrans (clearance based) transformation
 * for parameters.
 *
 * Nonmem events supported:
 *
 * evid = 0 observation line
 * evid = 1 dosing event - if rate>0 an infusion otherwise an IV bolus into the main cmt
 *
 * Needed input columns:
 * dv   observations
 * mdv  missing data flag (0 or 1)
 * evid event id
 * cmt  compartement (must be 1)
 * amt  administered dose
 * rate rate of input infusion
 *
 */
functions {
  /* calculate the absolute value of a - b in log-space with log(a)
     and log(b) given. Does so by squaring and taking the root, i.e.

     la = log(a)
     lb = log(b)

     sqrt( (a - b)^2 ) = sqrt( a^2 - 2 * a * b + b^2 )

     <=> 0.5 * log_diff_exp(log_sum_exp(2*la, 2*lb), log(2) + la + lb)
  */
  real log_diff_exp_abs(real la, real lb) {
    return(0.5 * log_diff_exp(log_sum_exp(2*la, 2*lb), log(2) + la + lb));
  }
  // same game for (a-b-c) = sqrt( a^2 + b^2 + c^2 + 2*bc - 2*ab - 2*ac  )
  real log_diff_exp_abs3(real la, real lb, real lc) {
    return(0.5 * log_diff_exp(log_sum_exp(log_sum_exp(log_sum_exp(2*la, 2*lb), 2*lc), log(2) + lb + lc), log(2) + log_sum_exp(la + lb, la + lc) ));
  }

  void pretty_print(real[,] x) {
    int d[2];
    d <- dims(x);
    for (m in 1:d[1]) {
      row_vector[d[2]] rv;
      for (n in 1:d[2])
	rv[n] <- round(1000*x[m,n])/1000.;
      print("row ", m, " = ", rv);      
    }
  }
  
  real[] twoCmtOral_ode(real t,
			real[] y,
			real[] theta,
			real[] x_r,
			int[] x_i) {
    real dydt[3];
    real ka;
    real k12;
    real k21;
    real k10;

    ka <- theta[1];
    k10 <- theta[2];
    k12 <- theta[3];
    k21 <- theta[4];
    
    dydt[1] <- -ka * y[1];
    dydt[3] <- k12 * y[2] - k21 * y[3];	    
    dydt[2] <- -dydt[1] - k10 * y[2] -dydt[3];
    
    return dydt;
  }
  
  /** ANALYTICAL SOLUTION:
   * Calculates the 2-cmt model for one patient given as input nonmem
   * type data and as input parameters the logarith of the micro rate
   * and micro constants
   *
   * possible events are infusions and IV bolus injections into the
   * main compartement
   *
   * returns the log-concentration of the central compartement
   **/
  real[,] ltwoCmtOralModel(real t0, real[] state0, real[] time, 
			   real lka, real lalphaR, real lbetaR, real lA, real lB) {
    real lstateRefOral; // ref state for the 2-cmt with oral cmt (only the oral cmt)
    real lstateRef[2];  // ref state for the 2-cmt without oral cmt
    real tref;
    int N;
    real alphaR;
    real betaR;
    real ka;
    real lAt;
    real lBt;
    real lstate[size(time),3];
    real lk12;
    real lk21;
    real lD;
    real lad2;
    real lbd2;
      real ltemp;

    N <- size(time);

    ka <- exp(lka);
    alphaR <- exp(lalphaR);
    betaR  <- exp(lbetaR);

    // Bateman coefficients
    lAt <- lA + lka - log_diff_exp_abs(lka, lalphaR);
    lBt <- lB + lka - log_diff_exp_abs(lka, lbetaR );

    // needed constant for the unobserved peripheral cmt C
    lD <- log_diff_exp(lalphaR, lbetaR);   // discriminant which is always positive
    ltemp <- log_sum_exp(lB + lD, lbetaR);
    lk12 <- log_diff_exp(log_sum_exp(lalphaR, lbetaR), log_sum_exp(2*ltemp, lalphaR + lbetaR) - ltemp );
    lk21 <- log_diff_exp(lalphaR, lA + lD);
      
    lad2 <- 2 * log_diff_exp_abs(lalphaR, lka);
    lbd2 <- 2 * log_diff_exp_abs(lbetaR , lka);

    //lC <- lk12 + lka - ( log_diff_exp(log_sum_exp(lalphaR + lbetaR, 2 * lka), lka + log_sum_exp(lalphaR, lbetaR) ) ) - lD;
    /*
    At <- exp(lalphaR+lbetaR) - (alphaR + betaR) * ka + exp(2*lka) ;
    C <- exp(lk12 + lka - lD) / (At ) ;
    Ct <- exp(lk12 + lka) / ( At ) ;
    */

    //( log_diff_exp(log_sum_exp(lalphaR + lbetaR, 2 * lka), lka + log_sum_exp(lalphaR, lbetaR) ) ) - lD;
    /**/

    /*
    print("ka = ", ka);
    print("alphaR = ", alphaR);
    print("betaR = ", betaR);
    print("lA = ", lA);
    print("lB = ", lB);
    print("lk12 = ", lk12);
    print("ltemp = ", ltemp);
    //print("C = ", C);
    //print("At = ", At);
    print("A2i = ", A2i);
    print("B2i = ", B2i);
    print("lk21 = ", lk21);
    //print("Ct = ", Ct);
    print("lD = ", lD);
    */

    // by convention time starts just at the first observation
    tref <- t0;
    lstateRefOral <- state0[1];
    lstateRef[1]  <- state0[2];
    lstateRef[2]  <- state0[3];
    for(i in 1:N) {
      real Dt;
      Dt <- time[i] - tref;

      lstate[i,1] <- lstateRefOral - ka * Dt;
      // solution for the concentration which is in the central and
      // peripheral cmt
      lstate[i,2] <- lstateRef[1] + log_sum_exp(lA - alphaR * Dt, lB - betaR * Dt);
      lstate[i,3] <- lstateRef[2] + log_sum_exp(lB - alphaR * Dt, lA - betaR * Dt);

      // other changes in the state can only meaningful be calculated
      // if Dt is large enough to allow for diffusion to occur
      if(time[i] != tref) {
	//lstate[i,2] <- log_sum_exp(lstate[i,2], lstateRef[2] - lk12 + log_diff_exp(log_sum_exp(lB + lalphaR - betaR * Dt, lA + lbetaR - alphaR * Dt), lk21 + log_sum_exp(lA - alphaR * Dt, lB - betaR * Dt)));
	//lstate[i,3] <- log_sum_exp(lstate[i,3], lstateRef[1] + lk12 - lD + log_diff_exp(-betaR * Dt, -alphaR * Dt));

	lstate[i,2] <- log_sum_exp(lstate[i,2], lstateRef[2] + lD - lk12 + lA + lB + log_diff_exp(- betaR * Dt, - alphaR * Dt) );
	lstate[i,3] <- log_sum_exp(lstate[i,3], lstateRef[1] + lk12 - lD + log_diff_exp(-betaR * Dt, -alphaR * Dt));

	
	// add in the part which stems from oral cmt which results in
	// the superposition of Bateman functions in the main cmt
	lstate[i,2] <- log_sum_exp(lstate[i,2], lstateRefOral + log_sum_exp(lAt + log_diff_exp_abs( -alphaR * Dt, -ka * Dt),
									    lBt + log_diff_exp_abs( -betaR  * Dt, -ka * Dt))
				   );
	// last, we add into the peripheral cmt the effect of the oral
	// cmt dosing
	//lstate[i,3] <- log_sum_exp(lstate[i,3], lstateRefOral + lk12 + lka - lD - ka * Dt + log( D*A2i*B2i + A2i * exp(- (alphaR-ka) * Dt) - B2i * exp(-(betaR-ka)*Dt)   ) );
	// the huge expression below is a sign-sorted version of (brute force)
	// k12 * ka /[ (alpha-ka) * (beta-ka) ] * [ exp(-ka * t) - (ka-beta)/(alpha-beta) * exp(-alpha*t) + (ka-alpha)/(alpha-beta) * exp(-beta*t) ] 
	lstate[i,3] <- log_sum_exp(lstate[i,3], lstateRefOral + lk12 + lka - lD - lad2 - lbd2 +
				   log_diff_exp(log_sum_exp(log_sum_exp(lD + log_sum_exp(lalphaR + lbetaR, 2*lka) - ka * Dt, lalphaR + lbd2 - alphaR * Dt), lka    + lad2 - betaR * Dt ),
						log_sum_exp(log_sum_exp(lD + lka + log_sum_exp(lalphaR,   lbetaR) - ka * Dt, lka     + lbd2 - alphaR * Dt), lbetaR + lad2 - betaR * Dt )
						)
				   );
      }

   }

    return lstate;
  }

  /*
   * transform from log of lka, CL, V1, Q, V2 to log of lka, alpha,
   * beta, A, B
   */
  vector trans(real lka, real lCL, real lV1, real lQ, real lV2) {
    vector[3] lk;
    real lkSum;
    vector[5] mm;
    // first project onto "k" parametrization and then onto micro
    // and macro constants

    // lk
    lk[1] <- lCL - lV1;
    // lk12
    lk[2] <- lQ - lV1;
    //lk21
    lk[3] <- lQ - lV2;

    // log(k+k12+k21)
    lkSum <- log_sum_exp(lk);

    mm[1] <- lka;

    // check that discriminat is for all patients real, i.e. that 
    // (k10 + k12 + k21)^2 - 4 k10 k21 > 0
    // otherwise the eigenvalues would be imaginary leading to oscillations
    if(2*lkSum < log(4.0) + lk[1] + lk[3])
      reject("System discriminant must be real!");

    // log of second rate constant beta
    mm[3] <- log(0.5) + log_diff_exp(lkSum, 0.5 * log_diff_exp(2*lkSum, log(4.0) + lk[1] + lk[3]) );

    // log of first rate constant alpha
    mm[2] <- lk[1] + lk[3] - mm[3];

    // macro constants
    mm[4] <- log_diff_exp_abs(lk[3], mm[2]) - log_diff_exp_abs(mm[2], mm[3]);
    mm[5] <- log_diff_exp_abs(lk[3], mm[3]) - log_diff_exp_abs(mm[3], mm[2]);

    return(mm);
  }

  vector trans_k(real lka, real lCL, real lV1, real lQ, real lV2) {
    vector[4] mm;

    mm[1] <- exp(lka);
    // lk10
    mm[2] <- exp(lCL - lV1);
    // lk12
    mm[3] <- exp(lQ - lV1);
    // lk21
    mm[4] <- exp(lQ - lV2);

    return(mm);
  }

  // the pk function is called for each patient and optionally at all
  // evid==2 event lines
  row_vector pk(row_vector theta, row_vector Xb, real time) {
    row_vector[num_elements(theta)] theta_j;

    theta_j <- theta;

    // weight with exponent 0.75 to clearance & Q
    theta_j[2] <- theta[2] +  0.75 * Xb[1];
    theta_j[4] <- theta[4] +  0.75 * Xb[1];
      
    // weight with exponent 1 to volume V1 & V2
    theta_j[3] <- theta[3] +  1.00 * Xb[1];
    theta_j[5] <- theta[5] +  1.00 * Xb[1];

    return(theta_j);
  }

  /*
   * evaluates the model function for all patients, making use of
   * ragged array structures.
   *
   * parameters for patient j are expected in the matrix params[J,4].
   * 
   * as input we assume the log of the parameters; these are
   * transformed to the micro rate and macro constants
   */
  vector evaluate_model(int[] id, matrix Xb, matrix theta_j,
			real[] time, real[] lamt, real[] rate, int[] evid, int[] cmt, int[] Nnext, real[] x_r, int[] x_i) {
    vector[size(id)] ly;
    int N;
    int j;
    int J;
    int Np;
    int tRef;
    real lstateRef[3];
    vector[5] theta;
    real lV[3];
    
    lV[1] <- 0;

    j <- 0;
    N <- size(id);
    J <- rows(Xb);
    Np <- cols(theta_j);

    for(i in 1:N) {
      
      // check for events which reset the current state
      if(id[i] != j  // new patient
	 ) {
	row_vector[Np] cov_theta_j;
	// reset state and grab new pk parameters
	tRef <- i;
	lstateRef <- rep_array(-25.0, 3);
	j <- id[i];
	// call pk function here? Yes, we should.
	cov_theta_j <- pk(theta_j[j], Xb[j], time[i]);
	theta <- trans(cov_theta_j[1], cov_theta_j[2], cov_theta_j[3], cov_theta_j[4], cov_theta_j[5]);
	lV[2] <- cov_theta_j[3];
	lV[3] <- cov_theta_j[5];
	/*
	print("Solving for patient j = ", j);
	print("theta_j[j] = ", theta_j[j]);
	print("theta = ", theta);
	print("Xb[j] = ", Xb[j]);
	*/
      }

      // consider optimization if solution is requested at two identical times?
      //if(i != 1 && id[i-1] == id[i] && time[i-1] == time[i]) {
      //tref <- i;
      //ly[i] <- lstateRef[2] - params[j,3];
      //} else

      if(time[tRef] == time[i]) {
	ly[i] <- lstateRef[cmt[i]] - lV[cmt[i]];
      } else {
	// run solver until we find the next dose or the end of the record
	real res[Nnext[i] + 1,3];
	res <- ltwoCmtOralModel(time[tRef],
				lstateRef,
				segment(time, i, Nnext[i] + 1),
				theta[1], theta[2], theta[3], theta[4], theta[5]
				);
	// copy over results for central cmt into final solution vector
	for(l in 0:Nnext[i]) {
	  ly[i+l] <- res[l+1,cmt[i]] - lV[cmt[i]];
	}
	// update new reference state and time
	lstateRef <- res[Nnext[i] + 1];
	i <- i + Nnext[i];
	tRef <- i;
      }
      
      // take care of events occuring, i.e. dosing events; currently
      // only IV shots are supported; infusions would require
      // super-position of homogeneous solution with non-homogeneous
      // distortion
      if(evid[i] == 1 && lamt[i] > -25.0) {
	//print("Injecting dose ", lamt[i], " into cmt = ", cmt[i]);
	lstateRef[cmt[i]] <- log_sum_exp(lstateRef[cmt[i]], lamt[i]);
      }
      
    }

    return ly;
  }

  vector evaluate_model_ode(int[] id, matrix Xb, matrix theta_j,
			    real[] time, real[] amt, real[] rate, int[] evid, int[] cmt, int[] Nnext, real[] x_r, int[] x_i) {
    vector[size(id)] ly;
    int N;
    int j;
    int J;
    int Np;
    int tRef;
    real stateRef[3];
    real theta[4];
    real lV[3];
    
    lV[1] <- 0;

    j <- 0;
    N <- size(id);
    J <- rows(Xb);
    Np <- cols(theta_j);

    for(i in 1:N) {
      
      // check for events which reset the current state
      if(id[i] != j  // new patient
	 ) {
	row_vector[Np] cov_theta_j;
	// reset state and grab new pk parameters
	tRef <- i;
	stateRef <- rep_array(1e-20, 3);
	j <- id[i];
	// call pk function here? Yes, we should.
	cov_theta_j <- pk(theta_j[j], Xb[j], time[i]);
	theta <- to_array_1d(trans_k(cov_theta_j[1], cov_theta_j[2], cov_theta_j[3], cov_theta_j[4], cov_theta_j[5]));
	lV[2] <- cov_theta_j[3];
	lV[3] <- cov_theta_j[5];
	/*
	print("Solving for patient j = ", j);
	print("theta_j[j] = ", theta_j[j]);
	print("theta = ", theta);
	print("Xb[j] = ", Xb[j]);
	*/
      }

      // consider optimization if solution is requested at two identical times?
      //if(i != 1 && id[i-1] == id[i] && time[i-1] == time[i]) {
      //tref <- i;
      //ly[i] <- lstateRef[2] - params[j,3];
      //} else

      if(time[tRef] == time[i]) {
	ly[i] <- log(stateRef[cmt[i]]) - lV[cmt[i]];
      } else {
	// run solver until we find the next dose or the end of the record
	real res[Nnext[i] + 1,3];
	/*
	print("Advancing by Nnext = ", Nnext[i] + 1);
	print("tRef = ", tRef);
	print("lstateRef = ", lstateRef);
	print("time = ", segment(time, i, Nnext[i] + 1));
	*/
	// integrate_ode call
	res <- integrate_ode(twoCmtOral_ode,
			     stateRef,
			     time[tRef],
			     segment(time, i, Nnext[i] + 1),
			     theta, x_r, x_i);

	// copy over results for central cmt into final solution vector
	for(l in 0:Nnext[i]) {
	  ly[i+l] <- log(res[l+1,cmt[i]]) - lV[cmt[i]];
	}
	// update new reference state and time
	stateRef <- res[Nnext[i] + 1];
	i <- i + Nnext[i];
	tRef <- i;
      }
      
      // take care of events occuring, i.e. dosing events; currently
      // only IV shots are supported; infusions would require
      // super-position of homogeneous solution with non-homogeneous
      // distortion
      if(evid[i] == 1 && amt[i] > 0) {
	//print("Injecting dose ", lamt[i], " into cmt = ", cmt[i]);
	//lstateRef[cmt[i]] <- log_sum_exp(lstateRef[cmt[i]], lamt[i]);
	stateRef[cmt[i]] <- stateRef[cmt[i]] + amt[i];
      }
      
    }

    return ly;
  }


  
  /*
   * subsets the input data structure to the indices given as second
   * argument
   */
  vector subset(vector cand, int[] ind_set) {
    vector[size(ind_set)] out;
    for(i in 1:size(ind_set))
      out[i] <- cand[ind_set[i]];
    return out;
  }
  matrix subset_matrix(matrix cand, int[] ind_set) {
    matrix[size(ind_set),cols(cand)] out;
    for(i in 1:size(ind_set))
      out[i] <- cand[ind_set[i]];
    return out;
  }
  int countObs(int[] cmt, int[] evid, int[] mdv, int tcmt) {
    int count;
    count <- 0;
    for(i in 1:size(cmt)) {
      if(cmt[i] == tcmt && evid[i] == 0 && mdv[i] == 0)
        count <- count + 1;
    }
    return count;
  }
}

data {
  int<lower = 1> N; // number of points where system needs to be calculated, i.e. observation time-points and dosing time-points; N >= O
  real<lower=0> time[N];
  vector<lower=0>[N] dv;  // observations
  real<lower=0> amt[N];
  real<lower=0> rate[N];
  int<lower=0> cmt[N];
  int<lower=0, upper=1> mdv[N];
  int<lower=0, upper=2> evid[N];
  int<lower=1> id[N];

  int<lower=0,upper=1> use_ode;

  // number of baseline covariates
  int<lower=0> BC;
  matrix[N,BC] base;

  // prior specification
  vector<lower=0>[5] prior_theta_gmean;
  vector<lower=0>[5] prior_theta_sd95;
  real<lower=0>      prior_sigma_sd95;
  real<lower=0>      prior_sigma_gmean;

  // random effects which are fit for those which have random effects
  // turned on
  int<lower=0,upper=1> random_effect[5];
  vector<lower=0>[sum(random_effect)] prior_omega_sd95;
  vector<lower=0>[sum(random_effect)] prior_omega_gmean;

  // tau for estimates of PK metrics
  real<lower=0> tau;
}

transformed data {
  vector[countObs(cmt, evid, mdv, 2)] ldv;
  int ind_obs[countObs(cmt, evid, mdv, 2)];
  int O;
  int J;
  int Nnext[N];        // counts for each entry the number of data lines until the next dose event
  real lamt[N];        // log dose
  int M[max(id)];            // number of observations per patient
  int MF[max(id)];           // first index of each patient in 1...N range
  matrix[max(id),BC] Xb;     // matrix of baseline covariates
  int E;
  int ind_eta[5];
  int  x_i[0];
  real x_r[0];

  // number of observations
  O <- countObs(cmt, evid, mdv, 2);

  // grab elements from dv which will enter the likelihood
  {
    int o;
    o <- 1;
    for(i in 1:N) {
      if(cmt[i] == 2 && mdv[i] == 0) {
	ind_obs[o] <- i;
	o <- o + 1;
      }
    }
  }

  ldv <- log(subset(dv, ind_obs));

  // number of patients
  J <- max(id);
  print("no of patients J = ", J);
  // build up auxilary ragged array structure indices
  // contains the number of entries in the data vector for a given
  // patient
  M <- rep_array(0, J);
  {
    int j;
    j <- 1;
    for(i in 1:N) {
      if(id[i] != j)
	j <- j + 1;
      M[j] <- M[j] + 1;
    }
  }
  print("avg no of records per patient = ", mean(to_vector(M)));

  MF <- rep_array(0, J); // first index of each patient, i.e. cum sum over M
  MF[1] <- 1;
  for(j in 2:J) {
    MF[j] <- MF[j-1] + M[j-1];
  }

  // setup base-line covariates matrix
  if(BC > 0)
    Xb <- subset_matrix(base, MF);

  for(b in 1:BC) {
    print("Mean of baseline covariate ", b, ": ", mean(col(Xb, b)));
  }

  // count the number of infusions and IV bolus events for reporting
  {
    vector[2] amt_tot;
    vector[2] Ndose;
    amt_tot <- rep_vector(0, 2);
    Ndose <- rep_vector(0, 2);
    for(i in 1:N) {
      if(evid[i] == 1 && amt[i] > 0) {
	int type;
	if(rate[i] > 0)
	  type <- 2;
	else
	  type <- 1;
	amt_tot[type] <- amt_tot[type] + amt[i];
	Ndose[type] <- Ndose[type] + 1.0;
      }
    }
    print("avg no/dose of IV boluses per patient: ", Ndose[1]/J, " / ", amt_tot[1]/Ndose[1] );
    print("avg no/dose of infusions  per patient: ", Ndose[2]/J, " / ", amt_tot[2]/Ndose[2] );
  }

  {
    int i;
    i <- 1;
    // random effects setup
    E <- sum(random_effect);
    ind_eta <- rep_array(0, 5);
    for(j in 1:5) {
      if(random_effect[j] == 1) {
	print("enabled random effect on theta[", j, "]")
	ind_eta[j] <- i;
	i <- i + 1;
      }
    }
  }

  // prepare a vector which counts for each patient always how many
  // observations the next dose is away, doing so by counting from the
  // back; if at the same time point first an observation and then a
  // dosing is scheduled, the counting is such that the dose is at the
  // observation (to stop the integrator there)
  {
    int j;
    int count;
    j <- -1;
    for(i in 1:N) {
      int k;
      k <- N-i+1;
      if(id[k] != j) {
	// new patient, reset
	count <- 0;
	j <- id[k];
      }
      // new dose, reset
      if(evid[k] == 1)
	count <- 0;
      // we just had a dose and the time-points are the same => make
      // this observation then count=0
      if(count == 1 && time[k] == time[k+1])
	count <- 0;
      Nnext[k] <- count;
      count <- count + 1;
    }
    
    // NOTE: Enabled stopping of integrator at each observation to
    // allow for time-changeing covariate
    //Nnext <- rep_array(0, N);
    //print("INFO: Assuming time-varying covariates and calling PK function with each record.");
  }

  // log-transform dose vector
  for(i in 1:N) {
    if(amt[i] > 0)
      lamt[i] <- log(amt[i]);
    else
      lamt[i] <- -30.;
  }

  // model becomes ill defined if no random effect at all is
  // requested
  if(E == 0)
    reject("At least one random effect must be defined.");
      
}

parameters {
  vector[5] theta; // log-population parameters 1=ka, 2=CL, 3=V1, 4=Q, 5=V2
  real<lower   = 0>    sigma;
  vector<lower=0>[E]   omega;
  matrix[J,E] xi_eta;
}

transformed parameters {
  vector[N] ipre;

  {
    matrix[J,5] theta_j;

    // setup parameter matrix
    for(j in 1:J) {
      for(k in 1:5) {
	if(random_effect[k] == 1)
	  theta_j[j,k] <- theta[k] + omega[ind_eta[k]] * xi_eta[j, ind_eta[k]];
	else
	  theta_j[j,k] <- theta[k];
      }
    }
    
    // evaluate model
    if(use_ode == 1) {
      ipre <- evaluate_model_ode(id, Xb, theta_j, time, amt, rate, evid, cmt, Nnext, x_r, x_i);
    } else {
      ipre <- evaluate_model(id, Xb, theta_j, time, lamt, rate, evid, cmt, Nnext, x_r, x_i);
    }
  }
}

model {
  theta ~ normal( log(prior_theta_gmean), log(prior_theta_sd95)/1.96 );
  sigma ~ lognormal(log(prior_sigma_gmean), log(prior_sigma_sd95)/1.96 );

  to_vector(xi_eta) ~ normal(0,1);
  omega ~ lognormal(log(prior_omega_gmean), log(prior_omega_sd95)/1.96 );

  // extract those elements which enter the likelihood for cmt=2,
  // which are the observed rows
  ldv ~ normal(subset(ipre, ind_obs), sigma);
}

generated quantities {
  real ka;
  real CL;
  real V1;
  real Q;
  real V2;
  real A;
  real B;
  real alphaR;
  real betaR;
  real Css;
  real AUCinf;
  vector[O] log_lik;

  ka <- exp(theta[1]);
  CL <- exp(theta[2]);
  V1 <- exp(theta[3]);
  Q  <- exp(theta[4]);
  V2 <- exp(theta[5]);

  {
    vector[5] lmicro;
    lmicro <- trans(theta[1], theta[2], theta[3], theta[4], theta[5]);
    alphaR <- exp(lmicro[2]);
    betaR <- exp(lmicro[3]);
    A <- exp(lmicro[4]);
    B <- exp(lmicro[5]);
  }

  Css <- A/(exp(alphaR * tau) - 1) + B/(exp(betaR * tau) - 1) - (A+B)/(exp(ka * tau) - 1);
  AUCinf <- A/alphaR + B/betaR - (A+B)/ka;

  // TODO: calculate tmax via a numerical approximation method; from
  // that get Cmax

  for(o in 1:O) log_lik[o] <- normal_log(ldv[o], ipre[ind_obs[o]], sigma);
}
