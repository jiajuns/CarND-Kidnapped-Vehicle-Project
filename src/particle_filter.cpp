/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 101;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i=0; i<num_particles; ++i){
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0f;
		particles.push_back(p);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double mean_x;
	double mean_y;
	double mean_theta;

	for (int i=0; i<num_particles; ++i){
		if (fabs(yaw_rate) < 0.00001){
      		mean_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      		mean_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
			mean_theta =  particles[i].theta;
		} else {
			mean_x = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			mean_y = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			mean_theta = particles[i].theta + yaw_rate*delta_t;
		}
		normal_distribution<double> dist_x(mean_x, std_pos[0]);
		normal_distribution<double> dist_y(mean_y, std_pos[1]);
		normal_distribution<double> dist_theta(mean_theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i=0; i<observations.size(); ++i){
		double obs_x = observations[i].x;
		double obs_y = observations[i].y;

		double small_dist = numeric_limits<double>::max();

		for (int j=0; j<predicted.size(); ++j){
			double lm_x = predicted[j].x;
			double lm_y = predicted[j].y;

			double temp_dist = dist(obs_x, obs_y, lm_x, lm_y);
			if (temp_dist < small_dist){
				small_dist = temp_dist;
				observations[i].id = predicted[j].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];

	for (int i=0; i<num_particles; ++i){

		vector<LandmarkObs> predictions;

		for (int s=0; s<map_landmarks.landmark_list.size(); ++s){

			double lm_x = map_landmarks.landmark_list[s].x_f;
			double lm_y = map_landmarks.landmark_list[s].y_f;
			int lm_id = map_landmarks.landmark_list[s].id_i;

			if (fabs(lm_x - particles[i].x) <= sensor_range && fabs(lm_y - particles[i].y) <= sensor_range){
				predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
			}
		}

		vector<LandmarkObs> transform_obs;
		for (int j=0; j<observations.size(); ++j){

			// transform into map coordinate system
			double x_obs = particles[i].x + cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y;
			double y_obs = particles[i].y + sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y;
			transform_obs.push_back(LandmarkObs{observations[j].id, x_obs, y_obs});
		}

		dataAssociation(predictions, transform_obs);
		particles[i].weight = 1.0f;

		for (int j=0; j<transform_obs.size(); ++j){

			double x_obs = transform_obs[j].x;
			double y_obs = transform_obs[j].y;
			int id_obs = transform_obs[j].id;

			double mu_x;
			double mu_y;

			for (int k=0; k<predictions.size(); ++k){
				if (id_obs == predictions[k].id){
					mu_x = predictions[k].x;
					mu_y = predictions[k].y;
				}
			}
			double gauss_norm = (1.0/(2 * M_PI * sig_x * sig_y));
			double exponent = (pow(x_obs - mu_x, 2))/(2 * pow(sig_x, 2)) + (pow(y_obs - mu_y, 2))/(2 * pow(sig_y, 2));
			particles[i].weight *= gauss_norm * exp(-exponent);
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	double max_w = 0.0;
	for (int i=0; i<particles.size(); i++){
		if (particles[i].weight > max_w){
			max_w = particles[i].weight;
		}
	}
	vector<Particle> particles3;
	default_random_engine gen;
  	uniform_real_distribution<double> distribution(0.0, 2*max_w);

	double beta = 0;
	int index = 0;
	for (int i=0; i<particles.size(); ++i){
		beta += distribution(gen);
		while(particles[index].weight < beta){
			beta -= particles[index].weight;
			index = (index + 1) % num_particles;
		}
		particles3.push_back(particles[index]);
	}
	particles = particles3;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
