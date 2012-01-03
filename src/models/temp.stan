parameters {
    double y[3,4];
}
derived parameters {
    double z[3,4];
    for (m in 1:3) for (n in 1:4) z[m,n] <- y[m,n];
}
model {
}