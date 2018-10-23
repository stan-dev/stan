functions{
  matrix covsqrt2corsqrt(matrix mat, int invert){ 
    int i = 1;
    int j = 2;
    matrix[rows(mat),cols(mat)] o;
    o=mat;
    o[i,j] = o[j,i];
    return o;
  }
}
