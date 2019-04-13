functions{
    real coupling(int t, int n, int N_node, matrix x1, matrix z, matrix K){ // sum( K * (x1_i - x1) )
        real s = 0.0;
        for(i in 1:N_node){
            s= s+ K[n,i]*(x1[i,t] - x1[n,t]);
        }
        return s;
    }
}
data {
    int<lower =1> N_time; 
    int<lower =1> N_node;
    matrix[N_node,N_time] x1_sim;
    matrix[N_node,N_node] K; // CM
    real dt;
    real tau;
}
transformed data{
    real I1 = 3.1;
}
parameters {
    vector<lower= -3.5, upper= -1.5>[N_node] x0; // m
    vector[N_node] x1_init;
    vector[N_node] z_init;
}

model {
    matrix[N_node,N_time] x1;
    matrix[N_node,N_time] z;

    x1[:,1] = x1_init;
    z[:,1] = z_init;

    for(t in 1:(N_time-1)){
        for(n in 1:N_node){
            x1[n,t+1] = x1[n,t] + dt * ( -(x1[n,t]^3) -2*(x1[n,t]^2) +1 - z[n,t] +I1);
            z[n,t+1] = z[n,t] + dt * tau* (4*(x1[n,t] - x0[n]) -z[n,t] -coupling(t,n,N_node, x1,z,K));
        }
    }
    
    //priors
    x1_init ~ normal(-2.0,0.1); 
    z_init ~ normal(4.0,0.1);
    for(n in 1:N_node){
        x0[n] ~ cauchy(-1.3, 1.0);
    }

    // Likelihood 
    for(i in 1:N_time){
        for(n in 1:N_node){
            x1_sim[n,i] ~ normal(x1[n,i], 1.0);
        }
    }
    
}
