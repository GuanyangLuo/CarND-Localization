/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  // 1000 is based on the Python Particle Filter example
  num_particles = 1000;  // TODO: Set the number of particles
  
  // TODO: (Guanyang Luo) add parameter checking
  double gps_x = x, gps_y = y, init_theta = theta; // First position from GPS
  double std_x = std[0], std_y = std[1], std_theta = std[2];  // Standard deviations for x, y, and theta

  // Create normal (Gaussian) distributions for x, y, and theta
  std::default_random_engine gen;
  normal_distribution<double> dist_x(gps_x, std_x);
  normal_distribution<double> dist_y(gps_y, std_y);
  normal_distribution<double> dist_theta(init_theta, std_theta);
  
  particles.clear();
  weights.clear();
  for (int i = 0; i < num_particles; i++) {
    // Add a particle from Gaussian distributions around the first position
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
    weights.push_back(p.weight);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // Create normal (Gaussian) distributions for noise
  // TODO: (Guanyang Luo) add parameter checking
  std::default_random_engine gen;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);
  
  // For each particle, predict the new state based on the bicycle model with Gaussian noise
  for (Particle& p : particles) {
    if (yaw_rate != 0.0){
      p.x = p.x + velocity / yaw_rate * ( sin(p.theta + yaw_rate * delta_t) - sin(p.theta) ) + dist_x(gen);
      p.y = p.y + velocity / yaw_rate * ( cos(p.theta) - cos(p.theta + yaw_rate * delta_t) ) + dist_y(gen); 
      p.theta = p.theta + yaw_rate * delta_t + dist_theta(gen);
    } else {
      p.x = p.x + velocity * delta_t * cos(p.theta) + dist_x(gen);
      p.y = p.y + velocity * delta_t * sin(p.theta) + dist_y(gen); 
      p.theta = p.theta + dist_theta(gen);
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  double min_dist = 0, distance = 0;
  
  for (LandmarkObs& o : observations) {
    // Initialize the association with a fake map landmark ID and a huge distance
    o.id = 0; //assuming the ID for map landmarks starts at 1
    min_dist = std::numeric_limits<double>::max();
    for (LandmarkObs p : predicted) {
      // Calculate the distance between the observation and map (predicted) landmark
      distance = dist(o.x, o.y, p.x, p.y);
      // Set the observation ID to that of the nearest map landmark
      if (distance < min_dist){
        min_dist = distance;
        o.id = p.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // TODO: (Guanyang Luo) add parameter checking
  
  double x_map = 0.0, y_map = 0.0, gauss_norm = 0.0, exponent = 0.0, weight = 1.0, final_weight = 1.0;
  int map_landmarks_index = 0;
       
  // Update the weight for each particle
  weights.clear();
  for (Particle& p : particles) {
    std::vector<Map::single_landmark_s> map_landmark_list = map_landmarks.landmark_list;
    
    // Optional for "blue lines" in simulator
    vector<int> associations;
    vector<double> sense_x; 
    vector<double> sense_y;
    
    // Transform the observations from the vehicle's coordinate to the map's coordinate, with respect to the particle
    vector<LandmarkObs> observations_by_particle;
    for (LandmarkObs o : observations) {
      // transform to map x and y coordinates
      x_map = p.x + (cos(p.theta) * o.x) - (sin(p.theta) * o.y);
      y_map = p.y + (sin(p.theta) * o.x) + (cos(p.theta) * o.y);
      LandmarkObs observations_p = {.id = 0, .x = x_map, .y = y_map};
      observations_by_particle.push_back(observations_p);
    }
    
    // Select the map landmarks that are within the particle's sensor range
    vector<LandmarkObs> predicted_by_particle;
    for (Map::single_landmark_s lm : map_landmark_list) {
      if (dist(p.x, p.y, lm.x_f, lm.y_f) <= sensor_range){
        LandmarkObs predicted_p = {.id = lm.id_i, .x = lm.x_f, .y = lm.y_f};
        predicted_by_particle.push_back(predicted_p);
      }      
    }
    
    // Perform Association for the observations and map landmarks
    dataAssociation(predicted_by_particle, observations_by_particle);      
   
    // Calculate and update the weight for the particle
    final_weight = 1.0;
    for (LandmarkObs op : observations_by_particle) {
      map_landmarks_index = op.id - 1; //assuming the ID for map landmarks starts at 1
      // calculate normalization term
      gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      // calculate exponent
      exponent = (pow(op.x - map_landmark_list[map_landmarks_index].x_f, 2) / (2 * pow(std_landmark[0], 2)))
                   + (pow(op.y - map_landmark_list[map_landmarks_index].y_f, 2) / (2 * pow(std_landmark[1], 2)));
      // calculate weight using normalization terms and exponent
      weight = gauss_norm * exp(-exponent);
      // update the final weight
      final_weight *= weight;
      
      // Optional for "blue lines" in simulator
      associations.push_back(op.id);
      sense_x.push_back(op.x);
      sense_y.push_back(op.y); 
    }
    p.weight = final_weight;
    weights.push_back(final_weight);
    
    // Optional for "blue lines" in simulator
    SetAssociations(p,associations,sense_x,sense_y);
  }
  
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Create a discrete distribution where the random integers correspond to indices
  // of the particle and the probability is defined by the particle's weight
  std::default_random_engine gen;
  std::discrete_distribution<> d(weights.begin(),weights.end());
  
  // Create new particles and weights vectors by sampling from the discrete distribution
  std::vector<Particle> resampled_particles;
  weights.clear();
  for (int i = 0; i < num_particles; i++) {
    Particle p = particles[d(gen)];
    resampled_particles.push_back(p);
    weights.push_back(p.weight);
  }
  particles = resampled_particles;
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}