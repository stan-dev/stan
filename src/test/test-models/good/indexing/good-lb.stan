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
model {
  y ~ normal(0, 1);
}
