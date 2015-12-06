/**
 * Check all possible legal indexing for assignment statements (left
 * side) and indexed expressions (right side).
 *
 * COMBINATORICS (m = multi-index, u = uni-index)
 * 
 * 1 index
 *    m
 *    u
 * 2 indexes
 *    uu
 *    um
 *    mu
 *    mm
 * 3 indexes
 *    uuu
 *    uum
 *    umu
 *    umm
 *    muu
 *    mum
 *    mmu
 *    mmm
 */
transformed data {
  real x;

  real a[4];
  real b[3];
  
  int c[4];
  int d[3];

  real e[5, 6];
  real f[3, 2];

  real g[3, 4, 5];
  real h[6, 7, 8];

  vector[3] q;
  vector[3] r;

  vector[4] s[5];
  vector[4] t[5];

  vector[4] u[5, 6];
  vector[4] v[5, 6];

  row_vector[3] q_rv;
  row_vector[3] r_rv;

  row_vector[4] s_rv[5];
  row_vector[4] t_rv[5];

  row_vector[4] u_rv[5, 6];
  row_vector[4] v_rv[5, 6];

  matrix[3, 4] aa;
  matrix[3, 4] bb;
  
  matrix[3, 4] cc[5];
  matrix[3, 4] dd[5];

  // int[]
  c[2:] <- d;
  d <- c[2:];
  c[3:] <- d[2:];
  d[2:] <- c[3:];

  // real[]
  a[2:] <- b;
  b <- a[2:];

  a[1] <- x;
  x <- a[1];

  // real[ , ]
  e[2:] <- f;
  f <- e[2:];

  e[2] <- a;
  a <- e[2];

  e[2:, 3:] <- f;
  f <- e[2:, 3:];

  e[2:, 1] <- a;
  a <- e[2:, 1];

  e[2, 3:] <- a;
  a <- e[2, 3:];

  e[2, 3] <- x;
  x <- e[2, 3];


  // real[ , , ]
  g[1] <- f;
  f <- g[1];

  g[2:] <- h;
  h <- g[2:];


  g[3, 4] <- d;
  a <- g[3, 4];

  g[1, 2:] <- f;
  f <- g[1, 2:];

  g[2:, 1] <- f;
  f <- g[2:, 1];

  g[2:, 3:] <- h;
  h <- g[2:, 3:];


  g[1, 2, 3] <- 5;
  x <- g[1, 2, 3];

  g[2, 3, 4:] <- d;
  a <- g[2, 3, 4:];

  g[2, 3:, 4] <- d;
  a <- g[2, 3:, 4];

  g[2:, 3, 4] <- d;  // assign from but not to int array
  a <- g[2:, 3, 4];

  g[2, 3:, 4:] <- f;
  f <- g[2, 3:, 4:];

  g[2:, 3, 4:] <- f;
  f <- g[2:, 3, 4:];

  g[2:, 3:, 4] <- f;
  f <- g[2:, 3:, 4];

  g[2:, 3:, 4:] <- h;
  h <- g[2:, 3:, 4:];

  // vector
  q[1] <- 1;
  
  q[1:] <- r;
  r <- q[1:];

  // vector[]
  s <- t;

  s[1] <- q;
  q <- s[1];

  s[1:] <- t;
  t <- s[1:];

  s[1, 1] <- x;
  x <- s[1, 1];

  s[1, 1:] <- q;
  q <- s[1, 2:];

  s[1:, 1] <- a;
  a <- s[1:, 1];

  s[1:, 2:] <- t;
  t <- s[1:, 2:];

  // vector[ , ]
  u <- v;

  u[1, 2, 3] <- x;
  x <- u[1, 2, 3];
  
  u[1, 2, 3:] <- q;
  q <- u[1, 2, 3:];

  u[1, 2:, 3] <- a;
  a <- u[1, 2:, 3];
  
  u[1:, 2, 3] <- a;
  a <- u[1:, 2, 3];

  u[1, 2:, 3:] <- s;
  s <-   u[1, 2:, 3:];
  
  u[2:, 1, 3:] <- s;
  s <- u[2:, 1, 3:];

  u[2:, 3:, 1] <- e;
  e <- u[2:, 3:, 1];
  
  u[1:, 2:, 3:] <- v;
  v <- u[1:, 2:, 3:];

  // row_vector
  q_rv[1] <- 1;
  
  q_rv[1:] <- r_rv;
  r_rv <- q_rv[1:];

  // row_vector[]
  s_rv <- t_rv;

  s_rv[1] <- q_rv;
  q_rv <- s_rv[1];

  s_rv[1:] <- t_rv;
  t_rv <- s_rv[1:];

  s_rv[1, 1] <- x;
  x <- s_rv[1, 1];

  s_rv[1, 1:] <- q_rv;
  q_rv <- s_rv[1, 2:];

  s_rv[1:, 1] <- a;
  a <- s_rv[1:, 1];

  s_rv[1:, 2:] <- t_rv;
  t_rv <- s_rv[1:, 2:];

  // row_vector[ , ]
  u_rv <- v_rv;

  u_rv[1, 2, 3] <- x;
  x <- u_rv[1, 2, 3];
  
  u_rv[1, 2, 3:] <- q_rv;
  q_rv <- u_rv[1, 2, 3:];

  u_rv[1, 2:, 3] <- a;
  a <- u_rv[1, 2:, 3];
  
  u_rv[1:, 2, 3] <- a;
  a <- u_rv[1:, 2, 3];

  u_rv[1, 2:, 3:] <- s_rv;
  s_rv <- u_rv[1, 2:, 3:];
  
  u_rv[2:, 1, 3:] <- s_rv;
  s_rv <- u_rv[2:, 1, 3:];

  u_rv[2:, 3:, 1] <- e;
  e <- u_rv[2:, 3:, 1];
  
  u_rv[1:, 2:, 3:] <- v_rv;
  v_rv <- u_rv[1:, 2:, 3:];

  // matrix
  aa <- bb;

  aa[1] <- q_rv;
  q_rv <- aa[1];

  aa[1, 2] <- x;
  x <- aa[1, 2];

  aa[2:] <- bb;
  bb <- aa[2:];

  aa[2:, 3] <- r;
  r <- aa[2:, 3];

  aa[2, 3:] <- r_rv;
  r_rv <- aa[2, 3:];

  aa[2:, 3:] <- bb;
  bb <- aa[2:, 3:];

  // matrix[]
  cc <- dd;

  cc[1] <- aa;
  aa <- cc[1];

  cc[1:] <- dd;
  dd <- cc[1:];

  cc[1, 2] <- q_rv;
  q_rv <- cc[1, 2];

  cc[2, 3:] <- aa;
  aa <- cc[2, 3:];

  cc[2:, 3] <- s_rv;
  s_rv <- cc[2:, 3];

  cc[2:, 3:] <- dd;
  dd <- cc[2:, 3:];

  cc[1, 2, 3] <- x;
  x <- cc[1, 2, 3];

  cc[1, 2, 3:] <- q_rv;
  q_rv <- cc[1, 2, 3:];
  
  cc[1, 2:, 3] <- q;
  q <- cc[1, 2:, 3];

  cc[2:, 3, 4] <- a;
  a <- cc[2:, 3, 4];

  cc[1, 2:, 3:] <- aa;
  aa <- cc[1, 2:, 3:];
  
  cc[1:, 2, 3:] <- s_rv;
  s_rv <- cc[1:, 2, 3:];

  cc[1:, 2:, 3] <- s;
  s <- cc[1:, 2:, 3];

  cc[1:, 2:, 3:] <- dd;
  dd <- cc[1:, 2:, 3:];
}
parameters {
  real y;
}
transformed parameters {
  // note: no int transformed params (i.e., no var_c, var_d)
  real var_x;

  real var_a[4];
  real var_b[3];
  
  real var_e[5, 6];
  real var_f[3, 2];

  real var_g[3, 4, 5];
  real var_h[6, 7, 8];

  vector[3] var_q;
  vector[3] var_r;

  vector[4] var_s[5];
  vector[4] var_t[5];

  vector[4] var_u[5, 6];
  vector[4] var_v[5, 6];

  row_vector[3] var_q_rv;
  row_vector[3] var_r_rv;

  row_vector[4] var_s_rv[5];
  row_vector[4] var_t_rv[5];

  row_vector[4] var_u_rv[5, 6];
  row_vector[4] var_v_rv[5, 6];

  matrix[3, 4] var_aa;
  matrix[3, 4] var_bb;
  
  matrix[3, 4] var_cc[5];
  matrix[3, 4] var_dd[5];


  // 1) ASSIGN DATA TO PARAMS  [see below for params to params]

  // real[]
  var_a[2:] <- b;
  var_b <- a[2:];

  var_a[1] <- x;
  var_x <- a[1];

  // real[ , ]
  var_e[2:] <- f;
  var_f <- e[2:];

  var_e[2] <- a;
  var_a <- e[2];

  var_e[2:, 3:] <- f;
  var_f <- e[2:, 3:];

  var_e[2:, 1] <- a;
  var_a <- e[2:, 1];

  var_e[2, 3:] <- a;
  var_a <- e[2, 3:];

  var_e[2, 3] <- x;
  var_x <- e[2, 3];


  // real[ , , ]
  var_g[1] <- f;
  var_f <- g[1];

  var_g[2:] <- h;
  var_h <- g[2:];

  var_g[3, 4] <- d;
  var_a <- g[3, 4];

  var_g[1, 2:] <- f;
  var_f <- g[1, 2:];

  var_g[2:, 1] <- f;
  var_f <- g[2:, 1];

  var_g[2:, 3:] <- h;
  var_h <- g[2:, 3:];

  var_g[1, 2, 3] <- 5;
  var_x <- g[1, 2, 3];

  var_g[2, 3, 4:] <- d;
  var_a <- g[2, 3, 4:];

  var_g[2, 3:, 4] <- d;
  var_a <- g[2, 3:, 4];

  var_g[2:, 3, 4] <- d;  // assign from but not to int array
  var_a <- g[2:, 3, 4];

  var_g[2, 3:, 4:] <- f;
  var_f <- g[2, 3:, 4:];

  var_g[2:, 3, 4:] <- f;
  var_f <- g[2:, 3, 4:];

  var_g[2:, 3:, 4] <- f;
  var_f <- g[2:, 3:, 4];

  var_g[2:, 3:, 4:] <- h;
  var_h <- g[2:, 3:, 4:];

  // // vector
  var_q[1] <- 1;
  
  var_q[1:] <- r;
  var_r <- q[1:];


  // vector[]
  var_s <- t;

  var_s[1] <- q;
  var_q <- s[1];

  var_s[1:] <- t;
  var_t <- s[1:];

  var_s[1, 1] <- x;
  var_x <- s[1, 1];

  var_s[1, 1:] <- q;
  var_q <- s[1, 2:];

  var_s[1:, 1] <- a;
  var_a <- s[1:, 1];

  var_s[1:, 2:] <- t;
  var_t <- s[1:, 2:];

  // vector[ , ]
  var_u <- v;

  var_u[1, 2, 3] <- x;
  var_x <- u[1, 2, 3];
  
  var_u[1, 2, 3:] <- q;
  var_q <- u[1, 2, 3:];

  var_u[1, 2:, 3] <- a;
  var_a <- u[1, 2:, 3];
  
  var_u[1:, 2, 3] <- a;
  var_a <- u[1:, 2, 3];

  var_u[1, 2:, 3:] <- s;
  var_s <-   u[1, 2:, 3:];
  
  var_u[2:, 1, 3:] <- s;
  var_s <- u[2:, 1, 3:];

  var_u[2:, 3:, 1] <- e;
  var_e <- u[2:, 3:, 1];
  
  var_u[1:, 2:, 3:] <- v;
  var_v <- u[1:, 2:, 3:];

  // row_vector
  var_q_rv[1] <- 1;
  
  var_q_rv[1:] <- r_rv;
  var_r_rv <- q_rv[1:];

  // row_vector[]
  var_s_rv <- t_rv;

  var_s_rv[1] <- q_rv;
  var_q_rv <- s_rv[1];

  var_s_rv[1:] <- t_rv;
  var_t_rv <- s_rv[1:];

  var_s_rv[1, 1] <- x;
  var_x <- s_rv[1, 1];

  var_s_rv[1, 1:] <- q_rv;
  var_q_rv <- s_rv[1, 2:];

  var_s_rv[1:, 1] <- a;
  var_a <- s_rv[1:, 1];

  var_s_rv[1:, 2:] <- t_rv;
  var_t_rv <- s_rv[1:, 2:];

  // row_vector[ , ]
  var_u_rv <- v_rv;

  var_u_rv[1, 2, 3] <- x;
  var_x <- u_rv[1, 2, 3];
  
  var_u_rv[1, 2, 3:] <- q_rv;
  var_q_rv <- u_rv[1, 2, 3:];

  var_u_rv[1, 2:, 3] <- a;
  var_a <- u_rv[1, 2:, 3];
  
  var_u_rv[1:, 2, 3] <- a;
  var_a <- u_rv[1:, 2, 3];
  
  var_u_rv[1, 2:, 3:] <- s_rv;
  var_s_rv <- u_rv[1, 2:, 3:];
  
  var_u_rv[2:, 1, 3:] <- s_rv;
  var_s_rv <- u_rv[2:, 1, 3:];

  var_u_rv[2:, 3:, 1] <- e;
  var_e <- u_rv[2:, 3:, 1];
  
  var_u_rv[1:, 2:, 3:] <- v_rv;
  var_v_rv <- u_rv[1:, 2:, 3:];

  // matrix
  var_aa <- bb;

  var_aa[1] <- q_rv;
  var_q_rv <- aa[1];

  var_aa[1, 2] <- x;
  var_x <- aa[1, 2];

  var_aa[2:] <- bb;
  var_bb <- aa[2:];

  var_aa[2:, 3] <- r;
  var_r <- aa[2:, 3];

  var_aa[2, 3:] <- r_rv;
  var_r_rv <- aa[2, 3:];

  var_aa[2:, 3:] <- bb;
  var_bb <- aa[2:, 3:];

  // matrix[]
  var_cc <- dd;
  var_cc[1] <- aa;
  var_aa <- cc[1];

  var_cc[1:] <- dd;
  var_dd <- cc[1:];

  var_cc[1, 2] <- q_rv;
  var_q_rv <- cc[1, 2];

  var_cc[2, 3:] <- aa;
  var_aa <- cc[2, 3:];

  var_cc[2:, 3] <- s_rv;
  var_s_rv <- cc[2:, 3];

  var_cc[2:, 3:] <- dd;
  var_dd <- cc[2:, 3:];

  var_cc[1, 2, 3] <- x;
  var_x <- cc[1, 2, 3];

  var_cc[1, 2, 3:] <- q_rv;
  var_q_rv <- cc[1, 2, 3:];
  
  var_cc[1, 2:, 3] <- q;
  var_q <- cc[1, 2:, 3];

  var_cc[2:, 3, 4] <- a;
  var_a <- cc[2:, 3, 4];

  var_cc[1, 2:, 3:] <- aa;
  var_aa <- cc[1, 2:, 3:];
  
  var_cc[1:, 2, 3:] <- s_rv;
  var_s_rv <- cc[1:, 2, 3:];

  var_cc[1:, 2:, 3] <- s;
  var_s <- cc[1:, 2:, 3];

  var_cc[1:, 2:, 3:] <- dd;
  var_dd <- cc[1:, 2:, 3:];

  // 2) ASSIGN PARAMS TO PARAMS  [see below for params to params]

  // real[]
  var_a[2:] <- var_b;
  var_b <- var_a[2:];

  var_a[1] <- var_x;
  var_x <- var_a[1];

  // real[ , ]
  var_e[2:] <- var_f;
  var_f <- var_e[2:];

  var_e[2] <- var_a;
  var_a <- var_e[2];

  var_e[2:, 3:] <- var_f;
  var_f <- var_e[2:, 3:];

  var_e[2:, 1] <- var_a;
  var_a <- var_e[2:, 1];

  var_e[2, 3:] <- var_a;
  var_a <- var_e[2, 3:];

  var_e[2, 3] <- var_x;
  var_x <- var_e[2, 3];


  // real[ , , ]
  var_g[1] <- var_f;
  var_f <- var_g[1];

  var_g[2:] <- var_h;
  var_h <- var_g[2:];

  var_a <- var_g[3, 4];

  var_g[1, 2:] <- var_f;
  var_f <- var_g[1, 2:];

  var_g[2:, 1] <- var_f;
  var_f <- var_g[2:, 1];

  var_g[2:, 3:] <- var_h;
  var_h <- var_g[2:, 3:];

  var_g[1, 2, 3] <- 5;
  var_x <- var_g[1, 2, 3];

  var_a <- var_g[2, 3, 4:];

  var_a <- var_g[2, 3:, 4];

  var_a <- var_g[2:, 3, 4];

  var_g[2, 3:, 4:] <- var_f;
  var_f <- var_g[2, 3:, 4:];

  var_g[2:, 3, 4:] <- var_f;
  var_f <- var_g[2:, 3, 4:];

  var_g[2:, 3:, 4] <- var_f;
  var_f <- var_g[2:, 3:, 4];

  var_g[2:, 3:, 4:] <- var_h;
  var_h <- var_g[2:, 3:, 4:];

  // vector
  var_q[1] <- 1;
  
  var_q[1:] <- var_r;
  var_r <- var_q[1:];


  // vector[]
  var_s <- var_t;

  var_s[1] <- var_q;
  var_q <- var_s[1];

  var_s[1:] <- var_t;
  var_t <- var_s[1:];

  var_s[1, 1] <- var_x;
  var_x <- var_s[1, 1];

  var_s[1, 1:] <- var_q;
  var_q <- var_s[1, 2:];

  var_s[1:, 1] <- var_a;
  var_a <- var_s[1:, 1];

  var_s[1:, 2:] <- var_t;
  var_t <- var_s[1:, 2:];

  // vector[ , ]
  var_u <- var_v;

  var_u[1, 2, 3] <- var_x;
  var_x <- var_u[1, 2, 3];
  
  var_u[1, 2, 3:] <- var_q;
  var_q <- var_u[1, 2, 3:];

  var_u[1, 2:, 3] <- var_a;
  var_a <- var_u[1, 2:, 3];
  
  var_u[1:, 2, 3] <- var_a;
  var_a <- var_u[1:, 2, 3];

  var_u[1, 2:, 3:] <- var_s;
  var_s <- var_u[1, 2:, 3:];
  
  var_u[2:, 1, 3:] <- var_s;
  var_s <- var_u[2:, 1, 3:];

  var_u[2:, 3:, 1] <- var_e;
  var_e <- var_u[2:, 3:, 1];
  
  var_u[1:, 2:, 3:] <- var_v;
  var_v <- var_u[1:, 2:, 3:];

  // // row_vector
  var_q_rv[1] <- 1;
  
  var_q_rv[1:] <- var_r_rv;
  var_r_rv <- var_q_rv[1:];

  // row_vector[]
  var_s_rv <- var_t_rv;

  var_s_rv[1] <- var_q_rv;
  var_q_rv <- var_s_rv[1];

  var_s_rv[1:] <- var_t_rv;
  var_t_rv <- var_s_rv[1:];

  var_s_rv[1, 1] <- var_x;
  var_x <- var_s_rv[1, 1];

  var_s_rv[1, 1:] <- var_q_rv;
  var_q_rv <- var_s_rv[1, 2:];

  var_s_rv[1:, 1] <- var_a;
  var_a <- var_s_rv[1:, 1];

  var_s_rv[1:, 2:] <- var_t_rv;
  var_t_rv <- var_s_rv[1:, 2:];

  // row_vector[ , ]
  var_u_rv <- var_v_rv;

  var_u_rv[1, 2, 3] <- var_x;
  var_x <- var_u_rv[1, 2, 3];
  
  var_u_rv[1, 2, 3:] <- var_q_rv;
  var_q_rv <- var_u_rv[1, 2, 3:];

  var_u_rv[1, 2:, 3] <- var_a;
  var_a <- var_u_rv[1, 2:, 3];
  
  var_u_rv[1:, 2, 3] <- var_a;
  var_a <- var_u_rv[1:, 2, 3];
  
  var_u_rv[1, 2:, 3:] <- var_s_rv;
  var_s_rv <- var_u_rv[1, 2:, 3:];
  
  var_u_rv[2:, 1, 3:] <- var_s_rv;
  var_s_rv <- var_u_rv[2:, 1, 3:];

  var_u_rv[2:, 3:, 1] <- var_e;
  var_e <- var_u_rv[2:, 3:, 1];
  
  var_u_rv[1:, 2:, 3:] <- var_v_rv;
  var_v_rv <- var_u_rv[1:, 2:, 3:];

  // matrix
  var_aa <- var_bb;

  var_aa[1] <- var_q_rv;
  var_q_rv <- var_aa[1];

  var_aa[1, 2] <- var_x;
  var_x <- var_aa[1, 2];

  var_aa[2:] <- var_bb;
  var_bb <- var_aa[2:];

  var_aa[2:, 3] <- var_r;
  var_r <- var_aa[2:, 3];

  var_aa[2, 3:] <- var_r_rv;
  var_r_rv <- var_aa[2, 3:];

  var_aa[2:, 3:] <- var_bb;
  var_bb <- var_aa[2:, 3:];

  // matrix[]
  var_cc <- var_dd;
  var_cc[1] <- var_aa;
  var_aa <- var_cc[1];

  var_cc[1:] <- var_dd;
  var_dd <- var_cc[1:];

  var_cc[1, 2] <- var_q_rv;
  var_q_rv <- var_cc[1, 2];

  var_cc[2, 3:] <- var_aa;
  var_aa <- var_cc[2, 3:];

  var_cc[2:, 3] <- var_s_rv;
  var_s_rv <- var_cc[2:, 3];

  var_cc[2:, 3:] <- var_dd;
  var_dd <- var_cc[2:, 3:];

  var_cc[1, 2, 3] <- var_x;
  var_x <- var_cc[1, 2, 3];

  var_cc[1, 2, 3:] <- var_q_rv;
  var_q_rv <- var_cc[1, 2, 3:];
  
  var_cc[1, 2:, 3] <- var_q;
  var_q <- var_cc[1, 2:, 3];

  var_cc[2:, 3, 4] <- var_a;
  var_a <- var_cc[2:, 3, 4];

  var_cc[1, 2:, 3:] <- var_aa;
  var_aa <- var_cc[1, 2:, 3:];
  
  var_cc[1:, 2, 3:] <- var_s_rv;
  var_s_rv <- var_cc[1:, 2, 3:];

  var_cc[1:, 2:, 3] <- var_s;
  var_s <- var_cc[1:, 2:, 3];

  var_cc[1:, 2:, 3:] <- var_dd;
  var_dd <- var_cc[1:, 2:, 3:];
} 
model {
  y ~ normal(0, 1);
}
