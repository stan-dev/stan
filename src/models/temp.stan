derived data {
    double w;
    {
        double c;
        c <- sqrt(2.0);
        w <- c;
    }
}

parameters {
    double y[3,4];
}
derived parameters {
    double z[3,4];
    { 
       double a;
       a <- y[1,1];
       z[1,1] <- a;
    }
}
model {
   double b;
   for (m in 1:3) for (n in 1:4) { b <- y[m,n] * 3; z[m,n] <- b; }
}