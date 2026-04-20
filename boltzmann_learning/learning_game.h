/*
 * @file            boltzmann_learning/learning_game.h
 * @description     
 * @author          nicewang <wangxiaonannice@gmail.com>
 * @createTime      2026-04-14
 * @lastModified    2026-04-21
 * Copyright Xiaonan (Nice) Wang. All rights reserved
*/

#ifndef LEARNING_GAME_H
#define LEARNING_GAME_H

#include "decision_maker.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <numeric>
#include <stdexcept>
#include <iomanip>
#include <limits>
#include <optional>
#include <string>

extern const bool DEBUG;

/**
 * @brief Class to learn no-regret policy
 * @tparam A The type representing an Action (e.g., int, string)
 * @tparam M The type representing a Measurement (e.g., int, string)
 */
template <typename A, typename M>
class LearningGame : public DecisionMaker<A, M> {
public:
    // Alias to easily use the ActionInfo struct defined in the abstract base class
    using ActionInfo = typename DecisionMaker<A, M>::ActionInfo;

    struct RegretInfo {
        double average_cost;        // average cost incurred
        double minimum_cost;        // minimum cost for stationary policy
        double cost_bound;          // theoretical bound on cost
        double regret;              // cost regret
        double regret_bound;        // theoretical bound on regret
                                    // regret_bound = alpha1 * minimum_cost + alpha0
        double normalization_sum;   // sum used to normalize "average cost".
        double alpha1;              // multiplicative factor for theoretical bound
        double alpha1_alpha0;       // additive factor for theoretical bound
    };

    /**
     * @brief Constructor for LearningGame class
     * @param action_set List of all possible actions.
     * Defaults to {}.
     * @param measurement_set List of all possible measurements.
     * @param finite_measurements indicates whether the set of measurements is discrete or continuous.
     * In the discrete case (True), the measurements will be elements of measurement_set.
     * In the continuous case (False), the measurements will be probabilities associated with the classes in
     * measurement_set.
     * @param decay_rate Exponential decay rate for information in units of 1/time.
     * Specifically, information decays as
     *                         exp(-decay_rate * time)
     *                         (0.0 means no decay).
     * Defaults to 0.0.
     * @param inverse_temperature Inverse of thermodynamic temperature
     * for the Boltzmann distribution.
     * Defaults to 0.01.
     * @param seed for action random number generator.
     * When None, a nondeterministic seed will be generated.
     * Defaults to None.
     * @param time_bound The max time between samples, only used to compute regret, though there are
     * other (better) ways to do so in general without this (i.e. compute the regret after one time step).
     * Defaults to 1.0.
     * @param compute_entropy
     *          True computes the entropy, 
     *          False sets it to nan. 
     *          Entropy for infinite measurement case is particularly slow to compute
     */
    LearningGame(
        vector<A> action_set,
        vector<M> measurement_set = {},
        bool finite_measurements = true,
        double decay_rate = 0.0,
        double inverse_temperature = 0.01,
        optional<unsigned int> seed = nullopt,
        double time_bound = 1.0,
        bool compute_entropy = true
    );

    /**
     * @brief Reset all parameters: total cost, min/max costs, energies
     */
    void reset();

    /**
     * @brief Returns a Boltzmann distribution
     * @param measurement which must be an element of measurement_set
     * @param time for the desired distribution. 
     * Defaults to 0.0.
     * @return A pair containing:
     * - probabilities (vector<double>): probability distribution
     * - entropy (doyble): distribution's entropy
     */
    pair<vector<double>, double> get_Boltzmann_distribution(
        const MeasurementInput<M>& measurement, 
        double time = 0.0
    );

    /**
     * @brief Gets (optimal) action for a given measurement. 
     * Overrides pure virtual from DecisionMaker.
     * @param measurement which must be an element of measurement_set
     * @param time for the desired distribution. 
     * Defaults to 0.0.
     * @return Optimal Action
     */
    ActionInfo get_action(
        const MeasurementInput<M>& measurement, 
        double time = 0.0
    ) override;

    /**
     * @brief Updates energies based on after-the-fact costs. 
     * Overrides pure virtual from DecisionMaker.
     * @param measurement which must be an element of measurement_set
     * @param costs for each action
     * @param time at which the measurement is made.
     * Defaults to 0.0.
     */
    void update_energies(
        const MeasurementInput<M>& measurement, 
        const map<A, double>& costs, 
        double time = 0.0
    ) override;

    /**
     * @brief Computes regret based on after-the-fact costs in update_energies()
     * @param display decides whether to display the regret (default: not display)
     * @return Regret
     */
    RegretInfo get_regret(bool display = false);

    /**
     * @brief Gets current energies
     * Energy associated with action a and measurements y,
     *      Each energy is of the form
     *          E[y][a,time_k] = \sum_{l=1}^k  exp(-lambda(time_k-time_l)) cost[y,a,time_l]
     * where t_k is the time at which we got the last update of the energies.
     */
    map<M, map<A, double>> get_energy() const;

private:
    vector<A> _action_set;
    size_t m_actions;
    vector<M> _measurement_set;
    bool finite_measurements;
    double decay_rate;
    double inverse_temperature;
    mt19937 rng;
    double time_bound;
    bool compute_entropy;

    map<M, map<A, double>> energy;
    double time_update;
    double total_cost;
    double normalization_sum;
    double min_cost;
    double max_cost;
};

/**
 * @brief Generate random integer for a given probability distribution
 * @param rng_engine random number generator
 * @param p vector with probabilities, which must add up to 1.0
 * @return random integer from 0 to len(p)-1
 */
int get_random_integer(mt19937& rng_engine, const vector<double>& p);

#endif // LEARNING_GAME_H