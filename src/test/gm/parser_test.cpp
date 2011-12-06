#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <istream>


#include "stan/gm/ast.hpp"
#include "stan/gm/parser.hpp"

TEST(GmParser, NormalExample) {
    stan::gm::program prog;
    
    std::ifstream normalExample;
    normalExample.open ("src/demo/normal_example/normal_example.stan");
    /*std::string testProg;
    testProg += "data {\n";
    testProg += "  int(0,) N;\n";
    testProg += "  double y[N];\n";
    testProg += "}\n";
    testProg += "parameters {\n";
    testProg += "  double mu;\n";
    testProg += "  double(0,) sigma;\n";
    testProg += "}\n";
    testProg += "model {\n";
    testProg += "  mu ~ normal(0,1);\n";
    testProg += "  sigma ~ cauchy(0,2);\n";
    testProg += "  for (n in 1:N) {\n";
    testProg += "    y[n] ~ normal(mu,sigma);\n";
    testProg += "  }\n";
    testProg += "}\n";

    std::istringstream testProgStream (testProg);
    bool succeeded = stan::gm::parse(testProgStream, "test1", prog); */
    bool succeeded = stan::gm::parse(normalExample, "NormalExample", prog);
    EXPECT_TRUE (succeeded);
}
