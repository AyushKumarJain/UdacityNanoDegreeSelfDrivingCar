#include "PID.h"

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {
  p_error = 0;
	i_error = 0;
	d_error = 0;
}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  Kp = Kp_;
  Ki = Ki_;
  Kd = Kd_;

}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
	i_error += cte;
  // We need to write p_error after calculating d_error, as we need old p_error in below formula
	d_error = cte - p_error;
	p_error = cte;

}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  return Kp * p_error + Ki * i_error + Kd * d_error;

  // return 0.0;  // TODO: Add your total error calc here!
}