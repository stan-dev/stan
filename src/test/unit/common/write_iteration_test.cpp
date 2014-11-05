#include <stan/common/write_iteration.hpp>
#include <gtest/gtest.h>
#include <test/test-models/good/common/test_lp.cpp>
#include <sstream>

typedef test_lp_model_namespace::test_lp_model Model;
typedef boost::ecuyer1988 rng_t;

class StanUi : public testing::Test {
public:
  void SetUp() {
    std::fstream empty_data_stream(std::string("").c_str());
    stan::io::dump empty_data_context(empty_data_stream);
    empty_data_stream.close();
    
    model_output.str("");
    model_ptr = new Model(empty_data_context, &model_output);
    base_rng.seed(123456);
  }
  
  void TearDown() {
    delete model_ptr;
  }
  
  rng_t base_rng;
  Model* model_ptr;
  std::stringstream model_output;
};

TEST_F(StanUi, write_iteration) {
  std::stringstream stream;
  Model model = *model_ptr;
  double lp;
  std::vector<double> cont_vector;
  std::vector<int> disc_vector;

  lp = 1.0;
  cont_vector.push_back(0);
  cont_vector.push_back(0);

  stan::common::write_iteration(stream, model, base_rng,
                            lp, cont_vector, disc_vector);
  EXPECT_EQ("1,0,0,1,1,2713\n", stream.str())
    << "the output should be (1,  0,       0,    exp(0),    exp(0), 2713) \n"
    << "                     (lp, y[1], y[2], exp(y[1]), exp(y[2]),  xgq)";

  EXPECT_EQ("", model_output.str());
}
