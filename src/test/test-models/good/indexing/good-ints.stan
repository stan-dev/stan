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

  int is[3];

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
  c[is] <- d;
  d <- c[is];
  c[is] <- d[is];
  d[is] <- c[is];

  // real[]
  a[is] <- b;
  b <- a[is];

  a[1] <- x;
  x <- a[1];

  // real[ , ]
  e[is] <- f;
  f <- e[is];

  e[2] <- a;
  a <- e[2];

  e[is, is] <- f;
  f <- e[is, is];

  e[is, 1] <- a;
  a <- e[is, 1];

  e[2, is] <- a;
  a <- e[2, is];

  e[2, 3] <- x;
  x <- e[2, 3];


  // real[ , , ]
  g[1] <- f;
  f <- g[1];

  g[is] <- h;
  h <- g[is];


  g[3, 4] <- d;
  a <- g[3, 4];

  g[1, is] <- f;
  f <- g[1, is];

  g[is, 1] <- f;
  f <- g[is, 1];

  g[is, is] <- h;
  h <- g[is, is];


  g[1, 2, 3] <- 5;
  x <- g[1, 2, 3];

  g[2, 3, is] <- d;
  a <- g[2, 3, is];

  g[2, is, 4] <- d;
  a <- g[2, is, 4];

  g[is, 3, 4] <- d;  // assign from but not to int array
  a <- g[is, 3, 4];

  g[2, is, is] <- f;
  f <- g[2, is, is];

  g[is, 3, is] <- f;
  f <- g[is, 3, is];

  g[is, is, 4] <- f;
  f <- g[is, is, 4];

  g[is, is, is] <- h;
  h <- g[is, is, is];

  // vector
  q[1] <- 1;
  
  q[is] <- r;
  r <- q[is];

  // vector[]
  s <- t;

  s[1] <- q;
  q <- s[1];

  s[is] <- t;
  t <- s[is];

  s[1, 1] <- x;
  x <- s[1, 1];

  s[1, is] <- q;
  q <- s[1, is];

  s[is, 1] <- a;
  a <- s[is, 1];

  s[is, is] <- t;
  t <- s[is, is];

  // vector[ , ]
  u <- v;

  u[1, 2, 3] <- x;
  x <- u[1, 2, 3];
  
  u[1, 2, is] <- q;
  q <- u[1, 2, is];

  u[1, is, 3] <- a;
  a <- u[1, is, 3];
  
  u[is, 2, 3] <- a;
  a <- u[is, 2, 3];

  u[1, is, is] <- s;
  s <-   u[1, is, is];
  
  u[is, 1, is] <- s;
  s <- u[is, 1, is];

  u[is, is, 1] <- e;
  e <- u[is, is, 1];
  
  u[is, is, is] <- v;
  v <- u[is, is, is];

  // row_vector
  q_rv[1] <- 1;
  
  q_rv[is] <- r_rv;
  r_rv <- q_rv[is];

  // row_vector[]
  s_rv <- t_rv;

  s_rv[1] <- q_rv;
  q_rv <- s_rv[1];

  s_rv[is] <- t_rv;
  t_rv <- s_rv[is];

  s_rv[1, 1] <- x;
  x <- s_rv[1, 1];

  s_rv[1, is] <- q_rv;
  q_rv <- s_rv[1, is];

  s_rv[is, 1] <- a;
  a <- s_rv[is, 1];

  s_rv[is, is] <- t_rv;
  t_rv <- s_rv[is, is];

  // row_vector[ , ]
  u_rv <- v_rv;

  u_rv[1, 2, 3] <- x;
  x <- u_rv[1, 2, 3];
  
  u_rv[1, 2, is] <- q_rv;
  q_rv <- u_rv[1, 2, is];

  u_rv[1, is, 3] <- a;
  a <- u_rv[1, is, 3];
  
  u_rv[is, 2, 3] <- a;
  a <- u_rv[is, 2, 3];

  u_rv[1, is, is] <- s_rv;
  s_rv <- u_rv[1, is, is];
  
  u_rv[is, 1, is] <- s_rv;
  s_rv <- u_rv[is, 1, is];

  u_rv[is, is, 1] <- e;
  e <- u_rv[is, is, 1];
  
  u_rv[is, is, is] <- v_rv;
  v_rv <- u_rv[is, is, is];

  // matrix
  aa <- bb;

  aa[1] <- q_rv;
  q_rv <- aa[1];

  aa[1, 2] <- x;
  x <- aa[1, 2];

  aa[is] <- bb;
  bb <- aa[is];

  aa[is, 3] <- r;
  r <- aa[is, 3];

  aa[2, is] <- r_rv;
  r_rv <- aa[2, is];

  aa[is, is] <- bb;
  bb <- aa[is, is];

  // matrix[]
  cc <- dd;

  cc[1] <- aa;
  aa <- cc[1];

  cc[is] <- dd;
  dd <- cc[is];

  cc[1, 2] <- q_rv;
  q_rv <- cc[1, 2];

  cc[2, is] <- aa;
  aa <- cc[2, is];

  cc[is, 3] <- s_rv;
  s_rv <- cc[is, 3];

  cc[is, is] <- dd;
  dd <- cc[is, is];

  cc[1, 2, 3] <- x;
  x <- cc[1, 2, 3];

  cc[1, 2, is] <- q_rv;
  q_rv <- cc[1, 2, is];
  
  cc[1, is, 3] <- q;
  q <- cc[1, is, 3];

  cc[is, 3, 4] <- a;
  a <- cc[is, 3, 4];

  cc[1, is, is] <- aa;
  aa <- cc[1, is, is];
  
  cc[is, 2, is] <- s_rv;
  s_rv <- cc[is, 2, is];

  cc[is, is, 3] <- s;
  s <- cc[is, is, 3];

  cc[is, is, is] <- dd;
  dd <- cc[is, is, is];
}
parameters {
  real y;
}
transformed parameters {
  // notis no int transformed params (i.e., no var_c, var_d)
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
  var_a[is] <- b;
  var_b <- a[is];

  var_a[1] <- x;
  var_x <- a[1];

  // real[ , ]
  var_e[is] <- f;
  var_f <- e[is];

  var_e[2] <- a;
  var_a <- e[2];

  var_e[is, is] <- f;
  var_f <- e[is, is];

  var_e[is, 1] <- a;
  var_a <- e[is, 1];

  var_e[2, is] <- a;
  var_a <- e[2, is];

  var_e[2, 3] <- x;
  var_x <- e[2, 3];


  // real[ , , ]
  var_g[1] <- f;
  var_f <- g[1];

  var_g[is] <- h;
  var_h <- g[is];

  var_g[3, 4] <- d;
  var_a <- g[3, 4];

  var_g[1, is] <- f;
  var_f <- g[1, is];

  var_g[is, 1] <- f;
  var_f <- g[is, 1];

  var_g[is, is] <- h;
  var_h <- g[is, is];

  var_g[1, 2, 3] <- 5;
  var_x <- g[1, 2, 3];

  var_g[2, 3, is] <- d;
  var_a <- g[2, 3, is];

  var_g[2, is, 4] <- d;
  var_a <- g[2, is, 4];

  var_g[is, 3, 4] <- d;  // assign from but not to int array
  var_a <- g[is, 3, 4];

  var_g[2, is, is] <- f;
  var_f <- g[2, is, is];

  var_g[is, 3, is] <- f;
  var_f <- g[is, 3, is];

  var_g[is, is, 4] <- f;
  var_f <- g[is, is, 4];

  var_g[is, is, is] <- h;
  var_h <- g[is, is, is];

  // // vector
  var_q[1] <- 1;
  
  var_q[is] <- r;
  var_r <- q[is];


  // vector[]
  var_s <- t;

  var_s[1] <- q;
  var_q <- s[1];

  var_s[is] <- t;
  var_t <- s[is];

  var_s[1, 1] <- x;
  var_x <- s[1, 1];

  var_s[1, is] <- q;
  var_q <- s[1, is];

  var_s[is, 1] <- a;
  var_a <- s[is, 1];

  var_s[is, is] <- t;
  var_t <- s[is, is];

  // vector[ , ]
  var_u <- v;

  var_u[1, 2, 3] <- x;
  var_x <- u[1, 2, 3];
  
  var_u[1, 2, is] <- q;
  var_q <- u[1, 2, is];

  var_u[1, is, 3] <- a;
  var_a <- u[1, is, 3];
  
  var_u[is, 2, 3] <- a;
  var_a <- u[is, 2, 3];

  var_u[1, is, is] <- s;
  var_s <-   u[1, is, is];
  
  var_u[is, 1, is] <- s;
  var_s <- u[is, 1, is];

  var_u[is, is, 1] <- e;
  var_e <- u[is, is, 1];
  
  var_u[is, is, is] <- v;
  var_v <- u[is, is, is];

  // row_vector
  var_q_rv[1] <- 1;
  
  var_q_rv[is] <- r_rv;
  var_r_rv <- q_rv[is];

  // row_vector[]
  var_s_rv <- t_rv;

  var_s_rv[1] <- q_rv;
  var_q_rv <- s_rv[1];

  var_s_rv[is] <- t_rv;
  var_t_rv <- s_rv[is];

  var_s_rv[1, 1] <- x;
  var_x <- s_rv[1, 1];

  var_s_rv[1, is] <- q_rv;
  var_q_rv <- s_rv[1, is];

  var_s_rv[is, 1] <- a;
  var_a <- s_rv[is, 1];

  var_s_rv[is, is] <- t_rv;
  var_t_rv <- s_rv[is, is];

  // row_vector[ , ]
  var_u_rv <- v_rv;

  var_u_rv[1, 2, 3] <- x;
  var_x <- u_rv[1, 2, 3];
  
  var_u_rv[1, 2, is] <- q_rv;
  var_q_rv <- u_rv[1, 2, is];

  var_u_rv[1, is, 3] <- a;
  var_a <- u_rv[1, is, 3];
  
  var_u_rv[is, 2, 3] <- a;
  var_a <- u_rv[is, 2, 3];
  
  var_u_rv[1, is, is] <- s_rv;
  var_s_rv <- u_rv[1, is, is];
  
  var_u_rv[is, 1, is] <- s_rv;
  var_s_rv <- u_rv[is, 1, is];

  var_u_rv[is, is, 1] <- e;
  var_e <- u_rv[is, is, 1];
  
  var_u_rv[is, is, is] <- v_rv;
  var_v_rv <- u_rv[is, is, is];

  // matrix
  var_aa <- bb;

  var_aa[1] <- q_rv;
  var_q_rv <- aa[1];

  var_aa[1, 2] <- x;
  var_x <- aa[1, 2];

  var_aa[is] <- bb;
  var_bb <- aa[is];

  var_aa[is, 3] <- r;
  var_r <- aa[is, 3];

  var_aa[2, is] <- r_rv;
  var_r_rv <- aa[2, is];

  var_aa[is, is] <- bb;
  var_bb <- aa[is, is];

  // matrix[]
  var_cc <- dd;
  var_cc[1] <- aa;
  var_aa <- cc[1];

  var_cc[is] <- dd;
  var_dd <- cc[is];

  var_cc[1, 2] <- q_rv;
  var_q_rv <- cc[1, 2];

  var_cc[2, is] <- aa;
  var_aa <- cc[2, is];

  var_cc[is, 3] <- s_rv;
  var_s_rv <- cc[is, 3];

  var_cc[is, is] <- dd;
  var_dd <- cc[is, is];

  var_cc[1, 2, 3] <- x;
  var_x <- cc[1, 2, 3];

  var_cc[1, 2, is] <- q_rv;
  var_q_rv <- cc[1, 2, is];
  
  var_cc[1, is, 3] <- q;
  var_q <- cc[1, is, 3];

  var_cc[is, 3, 4] <- a;
  var_a <- cc[is, 3, 4];

  var_cc[1, is, is] <- aa;
  var_aa <- cc[1, is, is];
  
  var_cc[is, 2, is] <- s_rv;
  var_s_rv <- cc[is, 2, is];

  var_cc[is, is, 3] <- s;
  var_s <- cc[is, is, 3];

  var_cc[is, is, is] <- dd;
  var_dd <- cc[is, is, is];

  // 2) ASSIGN PARAMS TO PARAMS  [see below for params to params]

  // real[]
  var_a[is] <- var_b;
  var_b <- var_a[is];

  var_a[1] <- var_x;
  var_x <- var_a[1];

  // real[ , ]
  var_e[is] <- var_f;
  var_f <- var_e[is];

  var_e[2] <- var_a;
  var_a <- var_e[2];

  var_e[is, is] <- var_f;
  var_f <- var_e[is, is];

  var_e[is, 1] <- var_a;
  var_a <- var_e[is, 1];

  var_e[2, is] <- var_a;
  var_a <- var_e[2, is];

  var_e[2, 3] <- var_x;
  var_x <- var_e[2, 3];


  // real[ , , ]
  var_g[1] <- var_f;
  var_f <- var_g[1];

  var_g[is] <- var_h;
  var_h <- var_g[is];

  var_a <- var_g[3, 4];

  var_g[1, is] <- var_f;
  var_f <- var_g[1, is];

  var_g[is, 1] <- var_f;
  var_f <- var_g[is, 1];

  var_g[is, is] <- var_h;
  var_h <- var_g[is, is];

  var_g[1, 2, 3] <- 5;
  var_x <- var_g[1, 2, 3];

  var_a <- var_g[2, 3, is];

  var_a <- var_g[2, is, 4];

  var_a <- var_g[is, 3, 4];

  var_g[2, is, is] <- var_f;
  var_f <- var_g[2, is, is];

  var_g[is, 3, is] <- var_f;
  var_f <- var_g[is, 3, is];

  var_g[is, is, 4] <- var_f;
  var_f <- var_g[is, is, 4];

  var_g[is, is, is] <- var_h;
  var_h <- var_g[is, is, is];

  // vector
  var_q[1] <- 1;
  
  var_q[is] <- var_r;
  var_r <- var_q[is];


  // vector[]
  var_s <- var_t;

  var_s[1] <- var_q;
  var_q <- var_s[1];

  var_s[is] <- var_t;
  var_t <- var_s[is];

  var_s[1, 1] <- var_x;
  var_x <- var_s[1, 1];

  var_s[1, is] <- var_q;
  var_q <- var_s[1, is];

  var_s[is, 1] <- var_a;
  var_a <- var_s[is, 1];

  var_s[is, is] <- var_t;
  var_t <- var_s[is, is];

  // vector[ , ]
  var_u <- var_v;

  var_u[1, 2, 3] <- var_x;
  var_x <- var_u[1, 2, 3];
  
  var_u[1, 2, is] <- var_q;
  var_q <- var_u[1, 2, is];

  var_u[1, is, 3] <- var_a;
  var_a <- var_u[1, is, 3];
  
  var_u[is, 2, 3] <- var_a;
  var_a <- var_u[is, 2, 3];

  var_u[1, is, is] <- var_s;
  var_s <- var_u[1, is, is];
  
  var_u[is, 1, is] <- var_s;
  var_s <- var_u[is, 1, is];

  var_u[is, is, 1] <- var_e;
  var_e <- var_u[is, is, 1];
  
  var_u[is, is, is] <- var_v;
  var_v <- var_u[is, is, is];

  // // row_vector
  var_q_rv[1] <- 1;
  
  var_q_rv[is] <- var_r_rv;
  var_r_rv <- var_q_rv[is];

  // row_vector[]
  var_s_rv <- var_t_rv;

  var_s_rv[1] <- var_q_rv;
  var_q_rv <- var_s_rv[1];

  var_s_rv[is] <- var_t_rv;
  var_t_rv <- var_s_rv[is];

  var_s_rv[1, 1] <- var_x;
  var_x <- var_s_rv[1, 1];

  var_s_rv[1, is] <- var_q_rv;
  var_q_rv <- var_s_rv[1, is];

  var_s_rv[is, 1] <- var_a;
  var_a <- var_s_rv[is, 1];

  var_s_rv[is, is] <- var_t_rv;
  var_t_rv <- var_s_rv[is, is];

  // row_vector[ , ]
  var_u_rv <- var_v_rv;

  var_u_rv[1, 2, 3] <- var_x;
  var_x <- var_u_rv[1, 2, 3];
  
  var_u_rv[1, 2, is] <- var_q_rv;
  var_q_rv <- var_u_rv[1, 2, is];

  var_u_rv[1, is, 3] <- var_a;
  var_a <- var_u_rv[1, is, 3];
  
  var_u_rv[is, 2, 3] <- var_a;
  var_a <- var_u_rv[is, 2, 3];
  
  var_u_rv[1, is, is] <- var_s_rv;
  var_s_rv <- var_u_rv[1, is, is];
  
  var_u_rv[is, 1, is] <- var_s_rv;
  var_s_rv <- var_u_rv[is, 1, is];

  var_u_rv[is, is, 1] <- var_e;
  var_e <- var_u_rv[is, is, 1];
  
  var_u_rv[is, is, is] <- var_v_rv;
  var_v_rv <- var_u_rv[is, is, is];

  // matrix
  var_aa <- var_bb;

  var_aa[1] <- var_q_rv;
  var_q_rv <- var_aa[1];

  var_aa[1, 2] <- var_x;
  var_x <- var_aa[1, 2];

  var_aa[is] <- var_bb;
  var_bb <- var_aa[is];

  var_aa[is, 3] <- var_r;
  var_r <- var_aa[is, 3];

  var_aa[2, is] <- var_r_rv;
  var_r_rv <- var_aa[2, is];

  var_aa[is, is] <- var_bb;
  var_bb <- var_aa[is, is];

  // matrix[]
  var_cc <- var_dd;
  var_cc[1] <- var_aa;
  var_aa <- var_cc[1];

  var_cc[is] <- var_dd;
  var_dd <- var_cc[is];

  var_cc[1, 2] <- var_q_rv;
  var_q_rv <- var_cc[1, 2];

  var_cc[2, is] <- var_aa;
  var_aa <- var_cc[2, is];

  var_cc[is, 3] <- var_s_rv;
  var_s_rv <- var_cc[is, 3];

  var_cc[is, is] <- var_dd;
  var_dd <- var_cc[is, is];

  var_cc[1, 2, 3] <- var_x;
  var_x <- var_cc[1, 2, 3];

  var_cc[1, 2, is] <- var_q_rv;
  var_q_rv <- var_cc[1, 2, is];
  
  var_cc[1, is, 3] <- var_q;
  var_q <- var_cc[1, is, 3];

  var_cc[is, 3, 4] <- var_a;
  var_a <- var_cc[is, 3, 4];

  var_cc[1, is, is] <- var_aa;
  var_aa <- var_cc[1, is, is];
  
  var_cc[is, 2, is] <- var_s_rv;
  var_s_rv <- var_cc[is, 2, is];

  var_cc[is, is, 3] <- var_s;
  var_s <- var_cc[is, is, 3];

  var_cc[is, is, is] <- var_dd;
  var_dd <- var_cc[is, is, is];
} 
model {
  y ~ normal(0, 1);
}
