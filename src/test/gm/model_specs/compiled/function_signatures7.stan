// data { 
//   int d_int;
//   int d_int_array[d_int];
//   real d_real;
//   real d_real_array[d_int];
//   matrix[d_int,d_int] d_matrix;
//   vector[d_int] d_vector;
//   row_vector[d_int] d_row_vector;
// }
// transformed data {
//   int transformed_data_int;
//   real transformed_data_real;
//   real transformed_data_real_array[d_int];
//   matrix[d_int,d_int] transformed_data_matrix;
//   vector[d_int] transformed_data_vector;
//   row_vector[d_int] transformed_data_row_vector;

//   transformed_data_int <- !transformed_data_int;
//   transformed_data_int <- !transformed_data_real;

//   transformed_data_int <- transformed_data_int && transformed_data_int;
//   transformed_data_int <- transformed_data_int && transformed_data_real;
//   transformed_data_int <- transformed_data_real && transformed_data_int;
//   transformed_data_int <- transformed_data_real && transformed_data_real;

//   transformed_data_int <- transformed_data_int || transformed_data_int;
//   transformed_data_int <- transformed_data_int || transformed_data_real;
//   transformed_data_int <- transformed_data_real || transformed_data_int;
//   transformed_data_int <- transformed_data_real || transformed_data_real;


//   transformed_data_int <- !transformed_data_real && transformed_data_real || !!!transformed_data_int;

//   transformed_data_int <- transformed_data_int < transformed_data_int;
//   transformed_data_int <- transformed_data_int < transformed_data_real;
//   transformed_data_int <- transformed_data_real < transformed_data_int;
//   transformed_data_int <- transformed_data_real < transformed_data_real;

//   transformed_data_int <- transformed_data_int <= transformed_data_int;
//   transformed_data_int <- transformed_data_int <= transformed_data_real;
//   transformed_data_int <- transformed_data_real <= transformed_data_int;
//   transformed_data_int <- transformed_data_real <= transformed_data_real;

//   transformed_data_int <- transformed_data_int > transformed_data_int;
//   transformed_data_int <- transformed_data_int > transformed_data_real;
//   transformed_data_int <- transformed_data_real > transformed_data_int;
//   transformed_data_int <- transformed_data_real > transformed_data_real;

//   transformed_data_int <- transformed_data_int >= transformed_data_int;
//   transformed_data_int <- transformed_data_int >= transformed_data_real;
//   transformed_data_int <- transformed_data_real >= transformed_data_int;
//   transformed_data_int <- transformed_data_real >= transformed_data_real;

//   transformed_data_int <- transformed_data_int == transformed_data_int;
//   transformed_data_int <- transformed_data_int == transformed_data_real;
//   transformed_data_int <- transformed_data_real == transformed_data_int;
//   transformed_data_int <- transformed_data_real == transformed_data_real;

//   transformed_data_int <- transformed_data_int != transformed_data_int;
//   transformed_data_int <- transformed_data_int != transformed_data_real;
//   transformed_data_int <- transformed_data_real != transformed_data_int;
//   transformed_data_int <- transformed_data_real != transformed_data_real;


// }
// parameters {
//   real p_real;
//   real p_real_array[d_int];
//   matrix[d_int,d_int] p_matrix;
//   vector[d_int] p_vector;
//   row_vector[d_int] p_row_vector;
// }
// transformed parameters {
//   real transformed_param_real;
//   real transformed_param_real_array[d_int];
//   matrix[d_int,d_int] transformed_param_matrix;
//   vector[d_int] transformed_param_vector;
//   row_vector[d_int] transformed_param_row_vector;

//   // using real results OK because of promotion

//   transformed_param_real <- !transformed_param_real;

//   transformed_param_real <- transformed_data_int && transformed_param_real;
//   transformed_param_real <- transformed_param_real && transformed_data_int;
//   transformed_param_real <- transformed_param_real && transformed_data_real;
//   transformed_param_real <- transformed_param_real && transformed_param_real;
//   transformed_param_real <- transformed_data_real && transformed_param_real;

//   transformed_param_real <- transformed_data_int || transformed_param_real;
//   transformed_param_real <- transformed_param_real || transformed_data_int;
//   transformed_param_real <- transformed_param_real || transformed_data_real;
//   transformed_param_real <- transformed_param_real || transformed_param_real;
//   transformed_param_real <- transformed_data_real || transformed_param_real;

//   transformed_param_real <- transformed_data_int <= transformed_param_real;
//   transformed_param_real <- transformed_param_real <= transformed_data_int;
//   transformed_param_real <- transformed_param_real <= transformed_data_real;
//   transformed_param_real <- transformed_param_real <= transformed_param_real;
//   transformed_param_real <- transformed_data_real <= transformed_param_real;

//   transformed_param_real <- transformed_data_int < transformed_param_real;
//   transformed_param_real <- transformed_param_real < transformed_data_int;
//   transformed_param_real <- transformed_param_real < transformed_data_real;
//   transformed_param_real <- transformed_param_real < transformed_param_real;
//   transformed_param_real <- transformed_data_real < transformed_param_real;

//   transformed_param_real <- transformed_data_int >= transformed_param_real;
//   transformed_param_real <- transformed_param_real >= transformed_data_int;
//   transformed_param_real <- transformed_param_real >= transformed_data_real;
//   transformed_param_real <- transformed_param_real >= transformed_param_real;
//   transformed_param_real <- transformed_data_real >= transformed_param_real;

//   transformed_param_real <- transformed_data_int > transformed_param_real;
//   transformed_param_real <- transformed_param_real > transformed_data_int;
//   transformed_param_real <- transformed_param_real > transformed_data_real;
//   transformed_param_real <- transformed_param_real > transformed_param_real;
//   transformed_param_real <- transformed_data_real > transformed_param_real;

//   transformed_param_real <- transformed_data_int == transformed_param_real;
//   transformed_param_real <- transformed_param_real == transformed_data_int;
//   transformed_param_real <- transformed_param_real == transformed_data_real;
//   transformed_param_real <- transformed_param_real == transformed_param_real;
//   transformed_param_real <- transformed_data_real == transformed_param_real;

//   transformed_param_real <- transformed_data_int != transformed_param_real;
//   transformed_param_real <- transformed_param_real != transformed_data_int;
//   transformed_param_real <- transformed_param_real != transformed_data_real;
//   transformed_param_real <- transformed_param_real != transformed_param_real;
//   transformed_param_real <- transformed_data_real != transformed_param_real;

//   // while (transformed_param_real) transformed_param_real <- 1.0;

//   while (transformed_data_real) {
//     transformed_param_real <- 1.0;
//     transformed_param_real <- 2.0;
//   }

//   while (transformed_data_real)
//     transformed_param_real <- 1.0;


//   if (transformed_param_real)
//     transformed_param_real <- 1.0;
//   else if (transformed_param_real)
//     transformed_param_real <- 2.0;
//   else if (transformed_param_real)
//     transformed_param_real <- 3.0;

//   if (transformed_param_real)
//     transformed_param_real <- 1.0;
//   else
//     transformed_param_real <- 2.0;

//   if (transformed_param_real) {
//     transformed_param_real <- 1.0;
//     transformed_param_real <- 1.0;
//   } else if (transformed_param_real) {
//     transformed_param_real <- 1.0;
//     transformed_param_real <- 1.0;
//   } else if (transformed_param_real) {
//     transformed_param_real <- 1.0;
//     transformed_param_real <- 1.0;
//   } else {
//     transformed_param_real <- 1.0;
//     transformed_param_real <- 1.0;
//   }    

//   if (transformed_data_int) {
//     transformed_param_real <- 1.0;
//     transformed_param_real <- 1.0;
//   } else if (transformed_data_int) {
//     transformed_param_real <- 1.0;
//     transformed_param_real <- 1.0;
//   } else if (transformed_data_int) {
//     transformed_param_real <- 1.0;
//     transformed_param_real <- 1.0;
//   } else {
//     transformed_param_real <- 1.0;
//     transformed_param_real <- 1.0;
//   }    

  

// }
model {  
}
