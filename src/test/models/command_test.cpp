/** 
 * This test tries out different command line options for a 
 * generated model.
 */
#include <gtest/gtest.h>
#include <stdexcept>

class ModelCommand : public ::testing::TestWithParam<int> {
public:
  static std::vector<std::string> arguments;
  static std::string model_path;

  static char get_path_separator() {
    char c;
    FILE *in;
    if(!(in = popen("make path_separator --no-print-directory", "r")))
      throw std::runtime_error("\"make path_separator\" has failed.");
    c = fgetc(in);
    pclose(in);
    return c;
  }
  
  
  void static SetUpTestCase() {
    arguments.push_back("help");
    arguments.push_back("data");
    arguments.push_back("init");
    arguments.push_back("samples");
    arguments.push_back("append_samples");
    arguments.push_back("seed");
    arguments.push_back("chain_id");
    arguments.push_back("iter");
    arguments.push_back("warmup");
    arguments.push_back("thin");
    arguments.push_back("refresh");
    arguments.push_back("leapfrog_steps");
    arguments.push_back("max_treedepth");
    arguments.push_back("epsilon");
    arguments.push_back("epsilon_pm");
    arguments.push_back("unit_mass_matrix");
    arguments.push_back("delta");
    arguments.push_back("gamma");
    arguments.push_back("test_grad");

    model_path.append("models");
    model_path.append(1, get_path_separator());
    model_path.append("command");
  }
};

std::vector<std::string> ModelCommand::arguments;
std::string ModelCommand::model_path;

TEST_F(ModelCommand, Arguments) {
  //  std::cout << "model_path: " << model_path << std::endl;
}

/*TEST_P(ModelCommand, PTest) {
std::cout << "PTest: " << GetParam() << std::endl;

}*/

//INSTANTIATE_TEST_CASE_P(abc, ModelCommand, testing::Range(1, 10));

