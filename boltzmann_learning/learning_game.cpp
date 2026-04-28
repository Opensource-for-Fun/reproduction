/*
 * @file            boltzmann_learning/learning_game.cpp
 * @description     LearningGame Implementation
 * @author          nicewang <wangxiaonannice@gmail.com>
 * @createTime      2026-04-25
 * @lastModified    2026-04-27
 * Copyright Xiaonan (Nice) Wang. All rights reserved
*/

#include "learning_game.h"

#include <limits>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <string>
#include <numeric>


// Define the extern DEBUG variable declared in the header
const bool DEBUG = false;

// ---------------------------------------------------------
// Helper Function
// ---------------------------------------------------------
int get_random_integer(std::mt19937& rng_engine, const std::vector<double>& p) {
    // std::discrete_distribution exactly mimics the cumulative sum 
    // and uniform random choice logic from the Python code.
    std::discrete_distribution<int> dist(p.begin(), p.end());
    return dist(rng_engine);
}

// ---------------------------------------------------------
// LearningGame
// ---------------------------------------------------------

template <typename A, typename M>
LearningGame<A, M>::LearningGame(
    std::vector<A> action_set,
    std::vector<M> measurement_set,
    bool finite_measurements,
    double decay_rate,
    double inverse_temperature,
    std::optional<std::uint32_t> seed,
    double time_bound,
    bool compute_entropy
) : 
    // parameters
    _action_set(action_set),
    finite_measurements(finite_measurements),
    decay_rate(decay_rate),
    inverse_temperature(inverse_temperature),
    time_bound(time_bound),
    compute_entropy(compute_entropy) 
{
    // parameters

    m_actions = _action_set.size();

    // If measurement set is not provided, initialize with a default constructed element
    if (measurement_set.empty()) {
        _measurement_set.push_back(M{}); 
    } else {
        _measurement_set = std::move(measurement_set);
    }
    // TODO: add check to make sure each list above has no duplicates

    // Initialize random number generator
    if (seed.has_value()) {
        rng.seed(seed.value());
    } else {
        std::random_device rd;
        rng.seed(rd());
    }

    reset();
}

template <typename A, typename M>
void LearningGame<A, M>::reset() {

    // initialize energies
    energy.clear();
    for (const auto& y : _measurement_set) {
        for (const auto& a : _action_set) {
            energy[y][a] = 0.0;
        }
    }

    // initialize time of last update for the energies
    time_update = 0.0;
    
    // initialize variables needed to compute regret
    total_cost = 0.0;

    normalization_sum = 0.0;

    // initialize bounds
    min_cost = std::numeric_limits<double>::infinity();  // minimum value of the cost
    max_cost = -std::numeric_limits<double>::infinity(); // maximum value of the cost
}

template <typename A, typename M>
std::pair<std::vector<double>, double> LearningGame<A, M>::get_Boltzmann_distribution(
    const MeasurementInput<M>& measurement_input, 
    double time
) {
    // compute Boltzmann distribution
    
    // From the equations it may not be immediately clear why the decay is included here. 
    // It is because at the time of selecting actions, 
    // we need to discount the energy due to time passed since we last updated our energy.
    //    |
    //    v
    // Decay for discounting the energy
    double decay = std::exp(-decay_rate * (time - time_update));
    
    vector<double> probabilities;
    double entropy = 0.0;

    if (finite_measurements) {
        if (!std::holds_alternative<M>(measurement_input)) {
            throw std::invalid_argument("finite_measurements is true, but map provided instead of single measurement.");
        }
        M measurement = std::get<M>(measurement_input);

        vector<double> energies_array;
        double min_energy = std::numeric_limits<double>::infinity();
        
        for (const auto& a : _action_set) {
            double e = decay * energy[measurement][a];
            energies_array.push_back(e);
            if (e < min_energy) {
                min_energy = e;
            }
        }

        // p = softmax(E, -\beta), beta: inverse_temperature -> Boltzmann Distribution
        std::vector<double> exponent;
        double total = 0.0;
        for (double e : energies_array) {
            double exp_val = -inverse_temperature * (e - min_energy);

            // ========== Mathematical Proof and Physical Interpretation ==========

            // (e - min_energy) (actually should be (e-min_energy)/decay) is regarded as the distance between (cur_action, opt_action)
            //      Specifically, energy could be regarded as cost:
            //                    min(energy) regarded as min(cost)
            //      the larger distance between cur_action and opt_action (i.e. E(cur) >> E(opt))
            //      the higher possibility this action will be disused gradually by Boltzmann-Learning
            //      thus, (e - min_energy) also corresponding to `Immediate Regret`

            // Bellman (Optimal) Equation: $$V(s) = \max_{a} [ R(s, a) + \gamma V(s') ]$$
            //          We regard energy as $$E = -V(s)$$, thus:
            //                      $$E = -V(s) = -\max_{a} [ R(s, a) + \gamma V(s') ]
            //                                  = -\gamma V(s') - \max_{a} [ R(s, a) ]
            //                                  = \gamma E - \max_{a} [ R(s, a) ]
            //                                  = \gamma E - \min_{a} [ cost(s, a) ]
            //                                  = \gamma E - \gamma \min_{a}[ E ]$$
            //          There is a trick:
            //                      Since we can not obtain the actual cost (i.e. J_{k}) when in get_Boltzmann_distribution, 
            //                              we use an approximation: "Regard E as cost" (mentioned before) -> use E instead of Cost            
            //          Thus, the transition function of ("faked") E is:
            //                      E = decay * (E - min_Energy) (i.e. e = decay * e, (decay = \gamma), e = e - min_e)

            // ========== Mathematical Proof and Physical Interpretation - End ==========

            exponent.push_back(exp_val);
            double p = std::exp(exp_val);
            probabilities.push_back(p);
            total += p;
        }

        // Compute entropy safely (even if some probabilities become zero)
        if (compute_entropy) {
            double dot_product = 0.0;
            for (std::size_t i = 0; i < probabilities.size(); i++) {
                dot_product += probabilities[i] * exponent[i];
            }
            entropy = -dot_product / total + std::log(total);
        } else {
            entropy = std::numeric_limits<double>::quiet_NaN();
        }

        // Normalize probability
        for (double& p : probabilities) {
            p /= total;
        }

    } else {
        if (!std::holds_alternative<std::map<M, double>>(measurement_input)) {
            throw srd::invalid_argument("finite_measurements is false, but single measurement provided instead of map.");
        }
        auto measurement = std::get<std::map<M, double>>(measurement_input);

        std::vector<double> measurement_values;
        double sum_meas = 0.0;
        for (const auto& m : _measurement_set) {
            double val = measurement.count(m) ? measurement.at(m) : 0.0;
            measurement_values.push_back(val);
            sum_meas += val;
        }

        if (std::abs(sum_meas - 1.0) > 1e-6) {
            throw std::invalid_argument("Measurement should sum to 1.0 with precision of 1e-6.");
        }

        // energies_array size: num_actions x num_measurements
        std::vector<std::vector<double>> energies_array(_action_set.size(), vector<double>(_measurement_set.size()));
        std::vector<double> min_energy(_measurement_set.size(), std::numeric_limits<double>::infinity());

        for (std::size_t a_idx = 0; a_idx < _action_set.size(); a_idx++) {
            for (std::size_t m_idx = 0; m_idx < _measurement_set.size(); m_idx++) {
                double e = decay * energy[_measurement_set[m_idx]][_action_set[a_idx]];
                energies_array[a_idx][m_idx] = e;
                if (e < min_energy[m_idx]) {
                    min_energy[m_idx] = e;
                }
            }
        }

        // $$P(a_{k} = a, c_{k} = c | F_{k}) =$$
        // $$exp(-\beta  E[a,c]) / \sum_{\bar{a},\bar{c}} exp(-\beta  E[\bar{a},\bar{c}])$$
        std::vector<std::vector<double>> pbar_a_c(_action_set.size(), std::vector<double>(_measurement_set.size()));
        double global_prob_sum = 0.0;

        for (std::size_t a_idx = 0; a_idx < _action_set.size(); a_idx++) {
            for (std::size_t m_idx = 0; m_idx < _measurement_set.size(); m_idx++) {
                double exp_val = -inverse_temperature * (energies_array[a_idx][m_idx] - min_energy[m_idx]);
                double p = exp(exp_val);
                pbar_a_c[a_idx][m_idx] = p;
                global_prob_sum += p;
            }
        }

        // Normalize pbar_a_c
        for (std::size_t a_idx = 0; a_idx < _action_set.size(); a_idx++) {
            for (std::size_t m_idx = 0; m_idx < _measurement_set.size(); m_idx++) {
                pbar_a_c[a_idx][m_idx] /= global_prob_sum;
            }
        }

        // P(a_k = a | F_k)
        probabilities.assign(_action_set.size(), 0.0);
        for (std::size_t a_idx = 0; a_idx < _action_set.size(); a_idx++) {
            for (std::size_t m_idx = 0; m_idx < _measurement_set.size(); m_idx++) {
                probabilities[a_idx] += pbar_a_c[a_idx][m_idx] * measurement_values[m_idx];
            }
        }

        double total = 0.0;
        for (double p : probabilities) {
            total += p;
        }

        if (total == 0.0) {
            std::cerr << "WARNING: Probabilities sum to 0. Assigning uniform distribution.\n";
            for (double& p : probabilities) {
                p = 1.0;
            }
            total = probabilities.size();
        }

        for (double& p : probabilities) {
            p /= total;
        }

        if (compute_entropy) {
            // Shannon Entropy
            entropy = 0.0;
            for (double p : probabilities) {
                if (p > 0.0) {
                    entropy -= p * std::log(p);
                }
            }
        } else {
            entropy = std::numeric_limits<double>::quiet_NaN();
        }
    }

    return {probabilities, entropy};
}

template <typename A, typename M>
typename LearningGame<A, M>::ActionInfo LearningGame<A, M>::get_action(const MeasurementInput<M>& measurement, double time) {
    auto [probabilities, entropy] = get_Boltzmann_distribution(measurement, time);
    
    std::discrete_distribution<std::size_t> dist(probabilities.begin(), probabilities.end());
    std::size_t action_index = dist(rng);
    
    return {_action_set[action_index], std::move(probabilities), entropy};
}

template <typename A, typename M>
void LearningGame<A, M>::update_energies(
    const MeasurementInput<M>& measurement, 
    const std::map<A, double>& costs, 
    double time
) {
    // update bounds
    for (const auto& a : _action_set) {
        double cost_val = costs.at(a);
        if (cost_val < min_cost) {
            min_cost = cost_val;
        }
        if (cost_val > max_cost) {
            max_cost = cost_val;
        }
    }

    double decay = std::exp(-decay_rate * (time - time_update));

    // update regrets
    double average_cost = 0.0;
    auto [probabilities, entropy] = get_Boltzmann_distribution(measurement, time);
    for (std::size_t k = 0; k < _action_set.size(); k++) {
        average_cost += probabilities[k] * costs.at(_action_set[k]);
    }
    total_cost = decay * total_cost + average_cost;

    // update energies
    for (const auto& m : _measurement_set) {
        double weight = 0.0;

        if (finite_measurements) {
            if (std::holds_alternative<M>(measurement) && m == std::get<M>(measurement)) {
                weight = 1.0;
            }
        } else {
            if (std::holds_alternative<std::map<M, double>>(measurement)) {
                auto meas_map = std::get<std::map<M, double>>(measurement);
                if (meas_map.count(m)) {
                    weight = meas_map.at(m);
                }
            }
        }

        // $$E_{a}(k+1) = e^{-\lambda \Delta t} \cdot (E_{a}(k) + J_{k})$$
        // Attention: we use cost to calculte E there, this differs from `get_Boltzmann_distribution`
        for (const auto& a : _action_set) {
            energy[m][a] = decay * (energy[m][a] + costs.at(a) * weight);
        }
    }

    // update normalization_sum
    //      W[time_k] = \sum_{l=1}^k  exp(-lambda(time_k-time_l))
    //                = 1 + exp(-lambda(time_k-time_{k-1})) W[time_{k-1}]
    normalization_sum = decay * normalization_sum + 1.0;

    // update time
    time_update = time;
}

template <typename A, typename M>
typename LearningGame<A, M>::RegretInfo LearningGame<A, M>::get_regret(bool display) {
    double inverse_decay = std::exp(decay_rate * time_bound);

    // Compute average cost
    double average_cost = 0.0;
    if (normalization_sum > 0) {
        average_cost = inverse_decay * total_cost / normalization_sum;
    }

    // Compute average minimum cost
    double minimum_cost = 0.0;
    for (const auto& m : _measurement_set) {
        double mn = std::numeric_limits<double>::infinity();
        for (const auto& a : _action_set) {
            if (energy[m][a] < mn) {
                mn = inverse_decay * energy[m][a];
            }
        }
        minimum_cost += mn;
    }

    if (normalization_sum > 0) {
        minimum_cost *= (inverse_decay / normalization_sum);
    }

    double regret = average_cost - minimum_cost;

    // Compute bounds
    double J0 = min_cost;
    double delta = 0.0;
    if (max_cost != min_cost) {
        delta = (std::exp(inverse_temperature * (J0 - min_cost)) - std::exp(inverse_temperature * (J0 - max_cost))) 
                / (inverse_temperature * (max_cost - min_cost));
    } else {
        delta = std::exp(inverse_temperature * (J0 - min_cost)); // Prevent division by 0
    }
    
    double delta0 = (std::exp(inverse_temperature * (J0 - min_cost)) - 1.0) / inverse_temperature - J0 + delta * min_cost;
    double alpha1 = 1.0 / delta;

    double cardinality_term;
    if (finite_measurements) {
        cardinality_term = _measurement_set.size() * std::log(_action_set.size());
    } else {
        cardinality_term = _measurement_set.size() * std::log(_measurement_set.size() * _action_set.size());
    }

    double alpha0 = delta0;
    if (normalization_sum > 0) {
        alpha0 = (cardinality_term * inverse_decay / inverse_temperature / normalization_sum) + delta0;
    }

    double cost_bound = alpha1 * (minimum_cost + alpha0);
    double regret_bound = cost_bound - minimum_cost;

    if (DEBUG) {
        std::cout << fixed << setprecision(6);
        std::cout << "  J0     = " << setw(10) << J0 
             << "  delta   = " << setw(10) << delta 
             << "  delta0 = " << setw(10) << delta0 << "\n";
        std::cout << "  alpha1 = " << setw(10) << alpha1 
             << "  alpha0 = " << setw(10) << alpha0 
             << "  alpha1*alpha0 = " << setw(10) << (alpha0 * alpha1) << "\n";
    }

    if (display) {
        std::cout << fixed << setprecision(6);
        std::cout << "  normalization_sum = " << setw(13) << normalization_sum 
             << "  alpha1        = " << setw(13) << alpha1
             << "  alpha0        = " << setw(13) << alpha0 << "\n";
        std::cout << "  minimum_cost      = " << setw(13) << minimum_cost 
             << "  average_cost = " << setw(13) << average_cost
             << "  cost_bound   = " << setw(13) << cost_bound << "\n";
        std::cout << "                                      regret       = " << setw(13) << regret
             << "  regret_bound = " << setw(13) << regret_bound << "\n";
    }

    return {average_cost, minimum_cost, cost_bound, regret, regret_bound, normalization_sum, alpha1, alpha1 * alpha0};
}

template <typename A, typename M>
std::map<M, std::map<A, double>> LearningGame<A, M>::get_energy() const {
    return energy;
}


// ====================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// 
// IMPORTANT: Since the template implementation is separated into a .cpp file,
// we MUST explicitly instantiate the templates for any types we plan 
// to use in the main executable. Otherwise, the linker will throw undefined 
// reference errors.
// ====================================================================

// Common Type Instantiations:
template class LearningGame<int, int>;
template class LearningGame<std::string, std::string>;
template class LearningGame<std::string, int>;
template class LearningGame<int, std::string>;
template class LearningGame<double, double>;