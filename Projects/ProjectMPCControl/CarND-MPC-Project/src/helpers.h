#ifndef HELPERS_H
#define HELPERS_H

#include <string>
#include "Eigen-3.3/Eigen/Core"
#include <math.h>

using Eigen::VectorXd;
using std::string;
using std::vector;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

//
// Helper functions to fit and evaluate polynomials.
//

// Evaluate a polynomial.
double polyeval(const VectorXd &coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); ++i) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from:
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
VectorXd polyfit(const VectorXd &xvals, const VectorXd &yvals, int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);

  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); ++i) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); ++j) {
    for (int i = 0; i < order; ++i) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);

  return result;
}

#endif  // HELPERS_H


Eigen::MatrixXd transGlobalToLocal(double x, double y, double psi, \
                vector<double> &pts_x, vector<double> &pts_y) {
  assert(pts_x.size() == pts_y.size());
  int len = pts_x.size();
  Eigen::MatrixXd loc_pts = Eigen::MatrixXd(2, len);

  for (int i = 0; i < len; i++) {
    loc_pts(0, i) =  cos(psi) * (pts_x[i] - x) + sin(psi) * (pts_y[i] - y);
    loc_pts(1, i) = -sin(psi) * (pts_x[i] - x) + cos(psi) * (pts_y[i] - y);  
  }

  return loc_pts;
}
// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }