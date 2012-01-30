#include <Eigen/Dense>

#include <iostream>
#include <vector>

#include <stan/maths/matrix.hpp>

int main(int argc, char* argv[]) {

  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::maths::get_base1;

  

  std::cout << "x[1]=" << get_base1(x,1,"x[1]",1) << std::endl;

  vector<double> y(2, 1.75);

  vector<vector<double> > xx(2);
  xx[0] = x;
  xx[1] = y;

  
  std::cout << "xx[2,1]=" 
            << get_base1(get_base1(xx,2,"xx[2]",1),
                         1,
                         "xx[2][1]",2)
            << std::endl;

  Matrix<double,Dynamic,Dynamic> m(2,3);
  m << 0, 1, 2, 3, 4, 5;
  std::cout << "m=" << m << std::endl;
  std::cout << "m.row(1)=" << m.row(1) << std::endl;
  
  std::cout << "get(m,2)=" << get_base1(m,2,"m[2]",1) << std::endl;
  std::cout << "get(m,2,3)=" << get_base1(m,2,3,"m[2][3]",2) << std::endl;

  try {
    get_base1(m,3,2,"foo",1);
  } catch (const std::r& e) {
    std::cout << e.what() << std::endl;
  }

}
