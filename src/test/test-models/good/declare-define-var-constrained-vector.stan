// no constrained types in functions or local variables

data {
  int d;
  simplex[d] d_simplex;
  simplex[d] d_simplex_ar[d];
  unit_vector[d] d_unit_vector;
  unit_vector[d] d_unit_vector_ar[d];
  ordered[d] d_ordered;
  ordered[d] d_ordered_ar[d];
  positive_ordered[d] d_positive_ordered;
  positive_ordered[d] d_positive_ordered_ar[d];
}
transformed data {
  simplex[d] td_simplex1 = d_simplex;
  simplex[d] td_simplex2 = d_simplex_ar[1];
  simplex[d] td_simplex_ar[d] = d_simplex_ar;

  unit_vector[d] td_unit_vector1 = d_unit_vector;
  unit_vector[d] td_unit_vector2 = d_unit_vector_ar[1];
  unit_vector[d] td_unit_vector_ar[d] = d_unit_vector_ar;

  ordered[d] td_ordered1 = d_ordered;
  ordered[d] td_ordered2 = d_ordered_ar[1];
  ordered[d] td_ordered_ar[d] = d_ordered_ar;

  positive_ordered[d] td_positive_ordered1 = d_positive_ordered;
  positive_ordered[d] td_positive_ordered2 = d_positive_ordered_ar[1];
  positive_ordered[d] td_positive_ordered_ar[d] = d_positive_ordered_ar;

  print("td_simplex1 = ", td_simplex1);
  print("td_simplex2 = ", td_simplex2);
  print("td_simplex_ar = ", td_simplex_ar);

  print("td_unit_vector1 = ", td_unit_vector1);
  print("td_unit_vector2 = ", td_unit_vector2);
  print("td_unit_vector_ar = ", td_unit_vector_ar);

  print("td_ordered1 = ", td_ordered1);
  print("td_ordered2 = ", td_ordered2);
  print("td_ordered_ar = ", td_ordered_ar);

  print("td_positive_ordered1 = ", td_positive_ordered1);
  print("td_positive_ordered2 = ", td_positive_ordered2);
  print("td_positive_ordered_ar = ", td_positive_ordered_ar);
}
transformed parameters {
  simplex[d] tp_simplex1 = d_simplex;
  simplex[d] tp_simplex2 = d_simplex_ar[1];
  simplex[d] tp_simplex_ar3[d] = d_simplex_ar;

  simplex[d] tp_simplex4 = tp_simplex1;
  simplex[d] tp_simplex5 = d_simplex_ar[1];
  simplex[d] tp_simplex_ar6[d] = tp_simplex_ar3;

  unit_vector[d] tp_unit_vector1 = d_unit_vector;
  unit_vector[d] tp_unit_vector2 = d_unit_vector_ar[1];
  unit_vector[d] tp_unit_vector_ar3[d] = d_unit_vector_ar;

  unit_vector[d] tp_unit_vector4 = tp_unit_vector1;
  unit_vector[d] tp_unit_vector5 = tp_unit_vector_ar3[2];
  unit_vector[d] tp_unit_vector_ar6[d] = tp_unit_vector_ar3;

  ordered[d] tp_ordered1 = d_ordered;
  ordered[d] tp_ordered2 = d_ordered_ar[1];
  ordered[d] tp_ordered_ar3[d] = d_ordered_ar;

  ordered[d] tp_ordered4 = tp_ordered1;
  ordered[d] tp_ordered5 = tp_ordered_ar3[3];
  ordered[d] tp_ordered_ar6[d] = tp_ordered_ar3;

  positive_ordered[d] tp_positive_ordered1 = d_positive_ordered;
  positive_ordered[d] tp_positive_ordered2 = d_positive_ordered_ar[1];
  positive_ordered[d] tp_positive_ordered_ar3[d] = d_positive_ordered_ar;

  positive_ordered[d] tp_positive_ordered4 = tp_positive_ordered1;
  positive_ordered[d] tp_positive_ordered5 = tp_positive_ordered_ar3[1];
  positive_ordered[d] tp_positive_ordered_ar6[d] = tp_positive_ordered_ar3;

  print("tp_simplex1 = ", tp_simplex1);
  print("tp_simplex2 = ", tp_simplex2);
  print("tp_simplex_ar3 = ", tp_simplex_ar3);
  print("tp_simplex4 = ", tp_simplex4);
  print("tp_simplex5 = ", tp_simplex5);
  print("tp_simplex_ar6 = ", tp_simplex_ar6);

  print("tp_unit_vector1 = ", tp_unit_vector1);
  print("tp_unit_vector2 = ", tp_unit_vector2);
  print("tp_unit_vector_ar3 = ", tp_unit_vector_ar3);
  print("tp_unit_vector4 = ", tp_unit_vector4);
  print("tp_unit_vector5 = ", tp_unit_vector5);
  print("tp_unit_vector_ar6 = ", tp_unit_vector_ar6);

  print("tp_ordered1 = ", tp_ordered1);
  print("tp_ordered2 = ", tp_ordered2);
  print("tp_ordered_ar3 = ", tp_ordered_ar3);
  print("tp_ordered4 = ", tp_ordered4);
  print("tp_ordered5 = ", tp_ordered5);
  print("tp_ordered_ar6 = ", tp_ordered_ar6);

  print("tp_positive_ordered1 = ", tp_positive_ordered1);
  print("tp_positive_ordered2 = ", tp_positive_ordered2);
  print("tp_positive_ordered_ar3 = ", tp_positive_ordered_ar3);
  print("tp_positive_ordered4 = ", tp_positive_ordered4);
  print("tp_positive_ordered5 = ", tp_positive_ordered5);
  print("tp_positive_ordered_ar6 = ", tp_positive_ordered_ar6);
}
model {
}
generated quantities {
  simplex[d] gq_simplex1 = d_simplex;
  simplex[d] gq_simplex2 = d_simplex_ar[1];
  simplex[d] gq_simplex_ar[d] = d_simplex_ar;

  unit_vector[d] gq_unit_vector1 = d_unit_vector;
  unit_vector[d] gq_unit_vector2 = d_unit_vector_ar[1];
  unit_vector[d] gq_unit_vector_ar[d] = d_unit_vector_ar;

  ordered[d] gq_ordered1 = d_ordered;
  ordered[d] gq_ordered2 = d_ordered_ar[1];
  ordered[d] gq_ordered_ar[d] = d_ordered_ar;

  positive_ordered[d] gq_positive_ordered1 = d_positive_ordered;
  positive_ordered[d] gq_positive_ordered2 = d_positive_ordered_ar[1];
  positive_ordered[d] gq_positive_ordered_ar[d] = d_positive_ordered_ar;

  print("gq_simplex1 = ", gq_simplex1);
  print("gq_simplex2 = ", gq_simplex2);
  print("gq_simplex_ar = ", gq_simplex_ar);

  print("gq_unit_vector1 = ", gq_unit_vector1);
  print("gq_unit_vector2 = ", gq_unit_vector2);
  print("gq_unit_vector_ar = ", gq_unit_vector_ar);

  print("gq_ordered1 = ", gq_ordered1);
  print("gq_ordered2 = ", gq_ordered2);
  print("gq_ordered_ar = ", gq_ordered_ar);

  print("gq_positive_ordered1 = ", gq_positive_ordered1);
  print("gq_positive_ordered2 = ", gq_positive_ordered2);
  print("gq_positive_ordered_ar = ", gq_positive_ordered_ar);
}
