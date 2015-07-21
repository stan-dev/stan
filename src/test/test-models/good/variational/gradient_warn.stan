data {
    int N;
    vector[N] p;
    int Ngrps;
    int<lower=1, upper=Ngrps> grp_index[N];
}
parameters {
    vector<lower=0.0001, upper=100>[Ngrps] sigmaGrp;
    vector<lower=-100, upper=1000>[Ngrps] muGrp;
}

model {
    int grpi;
    for (i in 1:N){
        grpi <- grp_index[i];
        p[i] ~ logistic(muGrp[grpi], sigmaGrp[grpi]);
    };
}
