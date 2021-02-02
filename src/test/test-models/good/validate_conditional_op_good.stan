functions {
  real foo(real fa_x, int fa_y) {
    real fl_x;
    int fl_y;
    fl_x = 1 ? fl_x : fl_y;
    fl_y = 1 ? fl_y : 0;
    fl_x = 1 ? fa_x : fl_x;
    fl_y = 1 ? fa_y : fl_y;
    return 2.0;
  }
}
data {
  int x;
  real y;
  array[2] real ya1;
  array[2, 2] real ya2;
  vector[5] z;
  array[2] vector[5] za1;
  array[2, 2] vector[5] za2;
  row_vector[6] w;
  array[2] row_vector[6] wa1;
  array[2, 2] row_vector[6] wa2;
  matrix[5, 6] m;
  array[2] matrix[5, 6] ma1;
  array[2, 2] matrix[5, 6] ma2;
}
transformed data {
  int tx;
  real ty;
  vector[5] tza;
  vector[5] tzb;
  vector[5] tzc;
  row_vector[6] twa;
  row_vector[6] twb;
  row_vector[6] twc;
  matrix[5, 6] tma;
  matrix[5, 6] tmb;
  matrix[5, 6] tmc;
  array[2] int tx1;
  array[2] real tya1;
  array[2] real tyb1;
  array[2] real tyc1;
  array[2] vector[5] tza1;
  array[2] vector[5] tzb1;
  array[2] row_vector[6] twa1;
  array[2] row_vector[6] twb1;
  array[2] row_vector[6] twc1;
  array[2] matrix[5, 6] tm1;
  array[2, 3] int txa2;
  array[2, 3] int txb2;
  array[2, 3] int txc2;
  array[2, 2] real tya2;
  array[2, 2] real tyb2;
  array[2, 2] vector[5] tza2;
  array[2, 2] vector[5] tzb2;
  array[2, 2] vector[5] tzc2;
  array[2, 2] row_vector[6] twa2;
  array[2, 2] row_vector[6] twb2;
  array[2, 2] matrix[5, 6] tma2;
  array[2, 2] matrix[5, 6] tmb2;
  array[2, 2] matrix[5, 6] tmc2;
  tx = 1 ? 2 : 3;
  ty = 1 ? 2.0 : 3.0;
  tx = x < 100 ? x : 100;
  ty = y > 100 ? 100 : y;
  ty = y < 100 ? y : 100;
  ty = y < 100 ? y : ty;
  tzc = x < 100 ? tza : tzb;
  twc = x < 100 ? twa : twb;
  tmc = x < 100 ? tma : tmb;
  tx1 = x < 100 ? txa2[1] : txb2[2];
  txc2 = x < 100 ? txa2 : txb2;
  tyc1 = x < 100 ? tya1 : tyb1;
  tya2 = x < 100 ? tya2 : tyb2;
  twc1 = x < 100 ? twa1 : twb1;
  twa2 = x < 100 ? twa2 : twb2;
  tm1 = x < 100 ? tma2[1] : tmb2[1];
  tma2 = x < 100 ? tma2 : tmb2;
  {
    real abcd;
    abcd = 1 ? abcd : 2.0;
  }
}
parameters {
  real py;
  vector[5] pz;
  row_vector[6] pw;
  matrix[5, 6] pm;
  array[2] real pya1;
  array[2, 2] real pya2;
  array[2] vector[5] pza1;
  array[2, 2] vector[5] pza2;
  array[2] matrix[5, 6] pma1;
  array[2, 2] matrix[5, 6] pma2;
}
transformed parameters {
  real tpy;
  vector[5] tpza;
  vector[5] tpzb;
  vector[5] tpzc;
  row_vector[6] tpwa;
  row_vector[6] tpwb;
  row_vector[6] tpwc;
  matrix[5, 6] tpma;
  matrix[5, 6] tpmb;
  matrix[5, 6] tpmc;
  array[2] real tpya1;
  array[2] real tpyb1;
  array[2] real tpyc1;
  array[2] vector[5] tpza1;
  array[2] vector[5] tpzb1;
  array[2] row_vector[6] tpwa1;
  array[2] row_vector[6] tpwb1;
  array[2] row_vector[6] tpwc1;
  array[2] matrix[5, 6] tpm1;
  array[2, 2] real tpya2;
  array[2, 2] real tpyb2;
  array[2, 2] vector[5] tpza2;
  array[2, 2] vector[5] tpzb2;
  array[2, 2] vector[5] tpzc2;
  array[2, 2] row_vector[6] tpwa2;
  array[2, 2] row_vector[6] tpwb2;
  array[2, 2] matrix[5, 6] tpma2;
  array[2, 2] matrix[5, 6] tpmb2;
  array[2, 2] matrix[5, 6] tpmc2;
  tpy = y < 100 ? x : y;
  tpy = y < 100 ? y : x;
  tpy = y < 100 ? y : py;
  tpy = y < 100 ? x : py;
  tpzc = x < 100 ? tpza : tpzb;
  tpwc = x < 100 ? tpwa : tpwb;
  tpmc = x < 100 ? tpma : tpmb;
  tpzc = x < 100 ? z : pz;
  tpzc = x < 100 ? pz : z;
  tpwc = x < 100 ? w : pw;
  tpwc = x < 100 ? pw : w;
  tpmc = x < 100 ? m : pm;
  tpmc = x < 100 ? pm : m;
  tpyc1 = ya1;
  tpyc1 = x < 100 ? tpya1 : tpyb1;
  tpyc1 = x < 100 ? ya1 : pya1;
  tpyc1 = x < 100 ? pya1 : ya1;
  tpya2 = x < 100 ? tpya2 : tpyb2;
  tpya2 = x < 100 ? ya2 : tpyb2;
  tpya2 = x < 100 ? tpya2 : ya2;
  tpwc1 = x < 100 ? tpwa1 : tpwb1;
  tpwc1 = x < 100 ? wa1 : tpwb1;
  tpwc1 = x < 100 ? tpwb1 : wa1;
  tpwa2 = x < 100 ? tpwa2 : tpwb2;
  tpwa2 = x < 100 ? wa2 : tpwb2;
  tpwa2 = x < 100 ? tpwb2 : wa2;
  tpm1 = x < 100 ? tpma2[1] : tpmb2[1];
  tpm1 = x < 100 ? ma2[1] : tpmb2[1];
  tpm1 = x < 100 ? tpmb2[1] : ma2[1];
  tpma2 = x < 100 ? tpma2 : tpmb2;
  tpma2 = x < 100 ? ma2 : pma2;
  tpma2 = x < 100 ? pma2 : ma2;
  {
    real abcde;
    abcde = 1 ? abcde : 2.0;
  }
}
model {
  py ~ normal(0, 1);
  {
    real abcdefg;
    abcdefg = 1 ? abcdefg : 2.0;
  }
}
generated quantities {
  int gqx;
  real gqy;
  vector[5] gqza;
  vector[5] gqzb;
  vector[5] gqzc;
  gqy = y < 100 ? x : y;
  gqy = y < 100 ? y : x;
  gqzc = x < 100 ? gqza : gqzb;
  {
    real abcdef;
    abcdef = 1 ? abcdef : 2.0;
  }
}

