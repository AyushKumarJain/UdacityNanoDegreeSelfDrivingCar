#ifndef PID_H
#define PID_H

// Student Code Start

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <stdlib.h>
#include <numeric>
using namespace std;
using std::vector;

// Student Code End

class PID {
 public:
  // Student Code Start
  /*
  * Errors
  */
  double d_p_error;
  double d_i_error;
  double d_d_error;

  /*
  * Coefficients
  */
  double d_Kp;
  double d_Kd;
  double d_Ki;

  // Variables to save best case
  double d_Kp_best;
  double d_Kd_best;
  double d_Ki_best;

  double d_total_error;
  double d_best_error;

  double d_abs_error;
  double d_squared_error;
  double d_mean_squared_error;

  // Twiddle parameters

  vector<double> vc_dKparam;
  vector<double> vc_dKtune;
  double norm_steer_value, dN_min_Tol;
  int  iTwd_iter, iN_max_steps, i_iter, k;

  int i_n_steps;
  int iPID_opt;
  int iSwitch_;
  double d_pre_cte;
  double d_steer_max;
  double dTol;

  double d_Speed_sum;

	// Flag to check, if the controller is initialized
	bool is_initialized;
  bool do_twiddle;


  // Student Code End



  /**
   * Constructor
   */
  PID();

  /**
   * Destructor.
   */
  virtual ~PID();

  /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_) The initial PID coefficients
   */
  void Init(double Kp_, double Ki_, double Kd_);

  /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   */
  void UpdateError(double cte);

  /**
   * Calculate the total PID error.
   * @output The total PID error
   */
  double TotalError();

  // Student Code Start
  /*
  * Twiddle algorithm to tune PID parameters
  */
  void Twiddle_param();

  /*
  * Restart Twiddle-loop
  */
  void Twiddle_reset();

  // private:
  // /**
  //  * PID Errors
  //  */
  // double p_error;
  // double i_error;
  // double d_error;

  // /**
  //  * PID Coefficients
  //  */ 
  // double Kp;
  // double Ki;
  // double Kd;

  // Student Code End

};

#endif  // PID_H