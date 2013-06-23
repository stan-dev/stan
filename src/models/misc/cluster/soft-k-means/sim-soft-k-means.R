K = 5;  # clusters
D = 8;  # dimensions
mu = array(rnorm(K*D),dim=c(K,D));

counts_per_cluster = rnbinom(K,20,0.4); 

N = sum(counts_per_cluster);

y = array(rnorm(N*D), dim=c(N,D));
index = 1;
for (k in 1:K) {
  for (n in 1:counts_per_cluster[k]) {
    for (d in 1:D) {
      y[index,d] = y[index,d] + mu[k,d];
    }
    index = index + 1;
  }
}

dump(c("N","D","K","y"),"sim-soft-k-means.data.R");

