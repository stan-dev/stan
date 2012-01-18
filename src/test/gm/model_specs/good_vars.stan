data {
    double a;
}
transformed data {
    double b;
    b <- 1.0; 
    b <- a + b + 1.0;
    {
        double b_local; // init to 0 locally
        b <- a + b + b_local;  
        b_local <- a + b + b_local;

    }
}
parameters {
    double c;
}
transformed parameters {
    double d; 
    d <- 1.0;
    d <- a + b + c + d;
    {
        double d_local;
        d <- a + b + c + d + d_local;
        d_local <- a + b + c + d + d_local;
    }
}
model {
    double e_local;
    e_local <- 1.0; 
    e_local <- a + b + c + d + e_local;
    {
       double f_local;
       f_local <- 1.0;
       e_local <- a + b + c + d + e_local + f_local;
       f_local <- a + b + c + d + e_local + f_local;
    }
}
generated quantities {
    double g;
    g <- 1.0;
    g <- a + b + c + d + g;
    {
        double g_local;
        g_local <- 1.0;
        g <- a + b + c + d + g + g_local;
        g_local <- a + b + c + d + g + g_local;
    }
}