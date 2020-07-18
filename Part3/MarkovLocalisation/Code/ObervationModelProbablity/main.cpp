#include <iostream>
#include "help_functions.h"

// norm_pdf(observation_measurement, pseudo_range_estimate, observation_stdev).

float value = 5.5; //11// TODO: assign a value, the difference in distances
float parameter = 5.0; //11// set as control parameter or observation measurement
float stdev = 1.0; // position or observation standard deviation

int main() {

  float prob = Helpers::normpdf(value, parameter, stdev);

  std::cout << prob << std::endl;

  return 0;
}

// Use the following with norm_pdf pressing "test run" to return each probability.
// float value = 5.5; //TODO: assign a value, the difference in distances
// float parameter = 5; //set as control parameter or observation measurement
// float stdev = 1.0; //position or observation standard deviation
// and

// float value = 11; //TODO: assign a value, the difference in distances
// float parameter = 11; //set as control parameter or observation measurement
// float stdev = 1.0; //position or observation standard deviation
// Result in vector form
// [3.99E-1,3.52E-1] Please note that grader allows any order and allows for slight differences in precision.