data {
  int a0;
}
transformed data {
  int td_a;
  int td_a1 = a0;
  int td_a2 = 4;
  {
    int loc_td_a;
    int loc_td_a1 = a0;
    int loc_td_a2 = 6;
  }
}
model {
  int model_a;
  int model_a1 = a0;
  {
    int loc_model_a;
    int loc_model_a1 = a0;
    int loc_model_a2 = 4;
  }
}
generated quantities {
  int gq_a;
  int gq_a1 = a0;
  int gq_a2 = 9;
  {
    int loc_gq_a;
    int loc_gq_a1 = a0;
    int loc_gq_a2 = 6;
  }
}
