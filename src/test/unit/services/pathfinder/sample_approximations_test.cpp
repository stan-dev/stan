#include <stan/services/pathfinder/single.hpp>
#include <gtest/gtest.h>


TEST(Pathfinder, calc_u_u2) {
  Eigen::VectorXd alpha(10);
  alpha << 0.99434035015686, 0.996596991589148, 0.996912934085565, 0.974233478220917, 0.985380229624499, 0.977804204019154, 0.963518774743505, 0.993435151076477, 1.01061395944603, 1.09215336785842;
  Eigen::MatrixXd Qk(10, 2);
  Eigen::VectorXd Qk_vec(20);
  Qk_vec << -0.278474679167602, -0.152157033895653, 0.0867505466957011, 0.482325877559475, -0.354786702113257, 0.471054430597708, 0.523769553707243, 0.192024657691747, -0.012178549697562, -0.00373742904594976
  , -0.0101587357838484, -0.00545544981545159, -0.00195803909699599, -0.00223400842625933, 0.0120665390403349, 0.00198288799603561, -0.00710942889371788, -0.0159165979874917, -0.34565269973477, -0.93804531304729;
  for (Eigen::Index i = 0; i < Qk_vec.size(); ++i) {
    Qk(i) = Qk_vec(i);
  }
  double logdetCholHk = 0.045802044684242;
  Eigen::VectorXd x_center(10);
  x_center << 0.148518589293209, 0.0837800375511616, -0.0363926200004713, -0.151859573239236, 0.116523128113017, -0.167778305381412, -0.119375573204488, -0.0521644831502672, 1.37484693636769, 0.635081893690443;
  Eigen::MatrixXd chol(2, 2);
  chol << 1.01074004693027, 0, 0.311492962362723, 1.04641139117945;
  chol.transposeInPlace();
  Eigen::VectorXd mock_rando_vec(10 * 10);
  mock_rando_vec << -0.560475646552213,-0.23017748948328,1.55870831414912,0.070508391424576,0.129287735160946,1.71506498688328,0.460916205989202,-1.26506123460653,-0.686852851893526,-0.445661970099958,1.22408179743946,0.359813827057364,0.400771450594052,0.11068271594512,-0.555841134754075,1.78691313680308,0.497850478229239,-1.96661715662964,0.701355901563686,-0.472791407727934,-1.06782370598685,-0.217974914658295,-1.02600444830724,-0.72889122929114,-0.625039267849257,-1.68669331074241,0.837787044494525,0.153373117836515,-1.13813693701195,1.25381492106993,0.426464221476814,-0.295071482992271,0.895125661045022,0.878133487533042,0.821581081637487,0.688640254100091,0.553917653537589,-0.0619117105767217,-0.305962663739917,-0.380471001012383,-0.694706978920513,-0.207917278019599,-1.26539635156826,2.16895596533851,1.20796199830499,-1.12310858320335,-0.402884835299076,-0.466655353623219,0.779965118336318,-0.0833690664718293,0.253318513994755,-0.028546755348703,-0.0428704572913161,1.36860228401446,-0.225770985659268,1.51647060442954,-1.54875280423022,0.584613749636069,0.123854243844614,0.215941568743973,0.379639482759882,-0.502323453109302,-0.33320738366942,-1.01857538310709,-1.07179122647558,0.303528641404258,0.448209778629426,0.0530042267305041,0.922267467879738,2.05008468562714,-0.491031166056535,-2.30916887564081,1.00573852446226,-0.709200762582393,-0.688008616467358,1.0255713696967,-0.284773007051009,-1.22071771225454,0.18130347974915,-0.138891362439045,0.00576418589988693,0.38528040112633,-0.370660031792409,0.644376548518833,-0.220486561818751,0.331781963915697,1.09683901314935,0.435181490833803,-0.325931585531227,1.14880761845109,0.993503855962119,0.54839695950807,0.238731735111441,-0.627906076039371,1.36065244853001,-0.600259587147127,2.18733299301658,1.53261062618519,-0.235700359100477,-1.02642090030678;
  Eigen::MatrixXd mock_rando(10, 10);
  for (Eigen::Index i = 0; i < mock_rando_vec.size(); ++i) {
    mock_rando(i) = mock_rando_vec(i);
  }
  auto mock_rando_gen = [&mock_rando]() {
    return mock_rando;
  };
  using stan::services::optimize::taylor_approx_t;
  using stan::services::optimize::calc_u_u2;
  taylor_approx_t tt{x_center, logdetCholHk, chol, Qk, false};
  auto xx = calc_u_u2(mock_rando_gen, tt, alpha);
  std::cout << "\n" << std::get<0>(xx) << "\n";
  std::cout << "\n" << std::get<1>(xx) << "\n";
}


TEST(Pathfinder, approximate_samples) {
  Eigen::VectorXd alpha(10);
  alpha << 0.99434035015686, 0.996596991589148, 0.996912934085565, 0.974233478220917, 0.985380229624499, 0.977804204019154, 0.963518774743505, 0.993435151076477, 1.01061395944603, 1.09215336785842;
  Eigen::MatrixXd Qk(10, 2);
  Eigen::VectorXd Qk_vec(20);
  Qk_vec << -0.278474679167602, -0.152157033895653, 0.0867505466957011, 0.482325877559475, -0.354786702113257, 0.471054430597708, 0.523769553707243, 0.192024657691747, -0.012178549697562, -0.00373742904594976
  , -0.0101587357838484, -0.00545544981545159, -0.00195803909699599, -0.00223400842625933, 0.0120665390403349, 0.00198288799603561, -0.00710942889371788, -0.0159165979874917, -0.34565269973477, -0.93804531304729;
  for (Eigen::Index i = 0; i < Qk_vec.size(); ++i) {
    Qk(i) = Qk_vec(i);
  }
  double logdetCholHk = 0.045802044684242;
  Eigen::VectorXd x_center(10);
  x_center << 0.148518589293209, 0.0837800375511616, -0.0363926200004713, -0.151859573239236, 0.116523128113017, -0.167778305381412, -0.119375573204488, -0.0521644831502672, 1.37484693636769, 0.635081893690443;
  Eigen::MatrixXd chol(2, 2);
  chol << 1.01074004693027, 0, 0.311492962362723, 1.04641139117945;
  chol.transposeInPlace();
  Eigen::VectorXd mock_rando_vec(10 * 10);
  mock_rando_vec << -0.560475646552213,-0.23017748948328,1.55870831414912,0.070508391424576,0.129287735160946,1.71506498688328,0.460916205989202,-1.26506123460653,-0.686852851893526,-0.445661970099958,1.22408179743946,0.359813827057364,0.400771450594052,0.11068271594512,-0.555841134754075,1.78691313680308,0.497850478229239,-1.96661715662964,0.701355901563686,-0.472791407727934,-1.06782370598685,-0.217974914658295,-1.02600444830724,-0.72889122929114,-0.625039267849257,-1.68669331074241,0.837787044494525,0.153373117836515,-1.13813693701195,1.25381492106993,0.426464221476814,-0.295071482992271,0.895125661045022,0.878133487533042,0.821581081637487,0.688640254100091,0.553917653537589,-0.0619117105767217,-0.305962663739917,-0.380471001012383,-0.694706978920513,-0.207917278019599,-1.26539635156826,2.16895596533851,1.20796199830499,-1.12310858320335,-0.402884835299076,-0.466655353623219,0.779965118336318,-0.0833690664718293,0.253318513994755,-0.028546755348703,-0.0428704572913161,1.36860228401446,-0.225770985659268,1.51647060442954,-1.54875280423022,0.584613749636069,0.123854243844614,0.215941568743973,0.379639482759882,-0.502323453109302,-0.33320738366942,-1.01857538310709,-1.07179122647558,0.303528641404258,0.448209778629426,0.0530042267305041,0.922267467879738,2.05008468562714,-0.491031166056535,-2.30916887564081,1.00573852446226,-0.709200762582393,-0.688008616467358,1.0255713696967,-0.284773007051009,-1.22071771225454,0.18130347974915,-0.138891362439045,0.00576418589988693,0.38528040112633,-0.370660031792409,0.644376548518833,-0.220486561818751,0.331781963915697,1.09683901314935,0.435181490833803,-0.325931585531227,1.14880761845109,0.993503855962119,0.54839695950807,0.238731735111441,-0.627906076039371,1.36065244853001,-0.600259587147127,2.18733299301658,1.53261062618519,-0.235700359100477,-1.02642090030678;
  Eigen::MatrixXd mock_rando(10, 10);
  for (Eigen::Index i = 0; i < mock_rando_vec.size(); ++i) {
    mock_rando(i) = mock_rando_vec(i);
  }
  auto mock_rando_gen = [&mock_rando]() {
    return mock_rando;
  };
  using stan::services::optimize::taylor_approx_t;
  using stan::services::optimize::approximation_samples;
  taylor_approx_t tt{x_center, logdetCholHk, chol, Qk, false};
  auto xx = approximation_samples(tt, 10, alpha, mock_rando_gen);
  Eigen::MatrixXd param_vals = xx;
  std::cout << "Params: \n" << param_vals << "\n";
  Eigen::RowVectorXd mean_vals = param_vals.colwise().mean();
  std::cout << "\n" << mean_vals << "\n";

  Eigen::RowVectorXd sd_vals = ((param_vals.rowwise() - mean_vals).array().square().matrix().colwise().sum().array() / (param_vals.rows() - 1)).sqrt();
  std::cout << "\n" << sd_vals << "\n";
}