functions{
  matrix covsqrt2corsqrt(matrix mat, int invert){ 
    matrix[rows(mat),cols(mat)] o;
    o=mat;
    o[1] = o[2];
    o[3:4] = o[1:2];
    return o;
  }
}
