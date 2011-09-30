data {
     int N;
     int NN[2];
     int NNN[3,NN[1]];

     int(1,2) n;
     int(,2) nn;
     int(1,) nnn;
     int(1,2) nnnn;
     int(2,3) n4[N];

     double y;
     double yy[2];
     double yyy[N,2];

     double(1.2,) z;
     double(,2.4) zz;
     double(1.2,2.4) zzz;
     double(,1) z4;
     double(N,12) z5[2,3];

     vector(N) u;
     vector(5) uu[2,3];

     row_vector(N) v;
     row_vector(5) vv[2,3];

     matrix(2,3) a;
     matrix(4,4) aa[7,2];

     simplex(N) t;
     simplex(5) tt[4];

     pos_ordered(N) w;
     pos_ordered(4) ww[2,4];

     corr_matrix(N) r;
     corr_matrix(2) rr[3,4];

     cov_matrix(N) s;
     cov_matrix(2) ss[3,2];
}
parameters {
     int(0,2) t_N;
     int(0,1) t_NN[2];
     int(12,28) t_NNN[3,NN[1]];

     int(1,2) t_n;
     int(2,3) t_n4[N];

     double t_y;
     double t_yy[2];
     double t_yyy[N,2];

     double(1.2,) t_z;
     double(,2.4) t_zz;
     double(1.2,2.4) t_zzz;
     double(,1) t_z4;
     double(N,12) t_z5[2,3];

     vector(N) t_u;
     vector(5) t_uu[2,3];
     
     row_vector(N) t_v;
     row_vector(5) t_vv[2,3];

     matrix(2,3) t_a;
     matrix(4,4) t_aa[7,2];

     simplex(N) t_t;
     simplex(5) t_tt[4];

     pos_ordered(N) t_w;
     pos_ordered(4) t_ww[2,4];

     corr_matrix(N) t_r;
     corr_matrix(2) t_rr[3,4];

     cov_matrix(N) t_s;
     cov_matrix(2) t_ss[3,2];
}
model {
   y ~ normal(0,1);
}
