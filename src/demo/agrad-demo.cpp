#include <vector>
#include <stan/agrad/agrad.hpp>

int main() {

  // tape function
  stan::agrad::var a = 0.8;
  stan::agrad::var b = 0.2;
  stan::agrad::var f = sin(a) + a * b;

  // compute gradient
  std::vector<stan::agrad::var> x;
  x.push_back(a);
  x.push_back(b);
  std::vector<double> g;
  f.grad(x,g);

  printf("found:    f=%f  df/da=%f  df/db=%f\n",
	 f.val(), g[0], g[1]);
  printf("expected: f=%f  df/da=%f  df/db=%f\n",
	 sin(0.8) + 0.8 * 0.2, 
	 cos(0.8) + 0.2,
	 0.8);
}
