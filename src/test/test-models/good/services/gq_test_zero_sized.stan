parameters {
  real a;
  vector[0] b;
  vector[5] c;
  array[0] vector[2] d;
  row_vector[6] e;
  array[2] vector[0] f;
  array[2] vector[3] g;
}
model {
  a ~ std_normal();
  c ~ std_normal();
  e ~ std_normal();
  g[1] ~ std_normal();
  g[2] ~ std_normal();
}

