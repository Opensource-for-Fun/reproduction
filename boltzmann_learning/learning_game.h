/*
 * @file            boltzmann_learning/learning_game.h
 * @description     
 * @author          nicewang <wangxiaonannice@gmail.com>
 * @createTime      2026-04-14
 * @lastModified    2026-04-27
 * Copyright Xiaonan (Nice) Wang. All rights reserved
*/

#ifndef LEARNING_GAME_H
#define LEARNING_GAME_H

#include "decision_maker.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <numeric>
#include <iomanip>
#include <optional>
#include <string>

extern const bool DEBUG;

/**
 * @brief Class to learn no-regret policy
 * @tparam A The type representing an Action (e.g., int, std::string)
 * @tparam M The type representing a Measurement (e.g., int, std::string)
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
     * @param measurement_set (optional) List of all possible measurements.
     * Defaults to {}.
     * @param finite_measurements (optional) indicates whether the set of measurements is discrete or continuous.
     * In the discrete case (True), the measurements will be elements of measurement_set.
     * In the continuous case (False), the measurements will be probabilities associated with the classes in
     * measurement_set.
     * Defaults to true.
     * @param decay_rate (optional) Exponential decay rate for information in units of 1/time. a.k.a lambda
     * Specifically, information decays as
     *                         exp(-decay_rate * time)
     *                         (0.0 means no decay).
     * Defaults to 0.0.
     * @param inverse_temperature (optional) Inverse of thermodynamic temperature
     * for the Boltzmann distribution.
     * Defaults to 0.01.
     * @param seed (optional) for action random number generator.
     * When None, a nondeterministic seed will be generated.
     * Defaults to None (nullopt).
     * @param time_bound The max time between samples, only used to compute regret, though there are
     * other (better) ways to do so in general without this (i.e. compute the regret after one time step).
     * Defaults to 1.0.
     * @param compute_entropy (optional)
     * True computes the entropy, 
     * False sets it to nan. 
     * Entropy for infinite measurement case is particularly slow to compute
     * Defaults to true.
     */
    LearningGame(
        std::vector<A> action_set,
        std::vector<M> measurement_set = {},
        bool finite_measurements = true,
        double decay_rate = 0.0,
        double inverse_temperature = 0.01,
        std::optional<std::uint32_t> seed = std::nullopt,
        double time_bound = 1.0,
        bool compute_entropy = true
    );

    /**
     * @brief Reset all parameters: total cost, min/max costs, energies
     */
    void reset();

    /**
     * @brief Returns a Boltzmann distribution
     * @param measurement which must be an element of _measurement_set
     * @param time (optional) for the desired distribution. 
     * Defaults to 0.0.
     * @return A pair containing:
     * - probabilities (vector<double>): probability distribution
     * - entropy (double): distribution's entropy
     */
    std::pair<std::vector<double>, double> get_Boltzmann_distribution(
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
        const std::map<A, double>& costs, 
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
     *      where t_k is the time at which we got the last update of the energies.
     */
    std::map<M, std::map<A, double>> get_energy() const;

private:
    std::vector<A> _action_set;              // set of all possible actions
    size_t m_actions;                        // length of action set
    std::vector<M> _measurement_set;         // set of all possible measurements
    bool finite_measurements;                // indicates whether the measurements are finite/discrete or continuous
    double decay_rate;                       // exponential decay rate for information in units of 1/time, a.k.a lambda
    double inverse_temperature;              // inverse of thermodynamic temperature for the Boltzmann distribution
    std::mt19937 rng;                        // random number generator to select actions
    double time_bound;                       // short term bound on the time gap between samples (referred to as \mu_0 in the paper)
    bool compute_entropy;

    std::map<M, std::map<A, double>> energy; // energy[y][a]: energy associated with action a and measurements y
                                             //      E[y][a,time_k] = \sum_{l=1}^k  exp(-lambda(time_k-time_l)) cost[y,a,time_l]
                                             //      where t_k is the time at which we got the last update of the energies.
    double time_update;                      // time at which the energies were updates last. 
                                             // Only needed when the information exponential decay rate `decay_rate` is not zero.
    double total_cost;                       // total cost incurred so far; used to compute regret
    double normalization_sum;                // normalization_sum: sum used to normalize "average cost". Given by
                                             //      W[time_k] = \sum_{l=1}^k  exp(-lambda(time_k-time_l))
                                             // where t_k is the time at which we got the last update of the energies.
                                             // For lambda=0.0 this is just the total number of updates.
    double min_cost;                         // minimum value of the cost
    double max_cost;                         // maximum value of the cost
};

/**
 * @brief Generate random integer for a given probability distribution
 * @param rng_engine random number generator
 * @param p vector with probabilities, which must add up to 1.0
 * @return random integer from 0 to len(p)-1
 */
int get_random_integer(std::mt19937& rng_engine, const std::vector<double>& p);

#endif // LEARNING_GAME_H