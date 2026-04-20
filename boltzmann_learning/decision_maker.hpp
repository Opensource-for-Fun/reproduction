/*
 * @file            boltzmann_learning/decision_maker.hpp
 * @description     DecisionMaker Abstract Template
 * @author          nicewang <wangxiaonannice@gmail.com>
 * @createTime      2026-04-14
 * @lastModified    2026-04-21
 * Copyright Xiaonan (Nice) Wang. All rights reserved
*/

#pragma once

#include <vector>
#include <map>
#include <variant>

using namespace std;

template <typename M>
using MeasurementInput = variant<M, map<M, double>>;

/**
 * @brief Abstract base class for decision makers
 * @tparam A The type representing an Action (e.g., int, string)
 * @tparam M The type representing a Measurement (e.g., int, string)
 */
template <typename A, typename M>
class DecisionMaker {
public:
    // Struct for returning multiple values from get_action.
    struct ActionInfo {
        A action;                           // Action itself
        vector<double> probabilities;  // Probability distribution used to select action
        double entropy;                     // Distribution's entropy
    };

    /**
     * @brief Virtual destructor is required for abstract base classes 
     * to ensure proper cleanup of derived objects.
     */
    virtual ~DecisionMaker() = default;

    /**
     * @brief Gets (optimal) action for a given measurement
     * @param measurement Measurement input
     * @param time Time for the desired action. Defaults to 0.0.
     * @return ActionInfo struct containing action, probabilities, and entropy
     */
    virtual ActionInfo get_action(const MeasurementInput<M>& measurement, double time = 0.0) = 0;

    /**
     * @brief Updates energies based on after-the-fact costs
     * @param measurement Measurement input
     * @param costs Costs mapping for each action
     * @param time The time at which the measurement is made. Defaults to 0.0.
     */
    virtual void update_energies(
        const MeasurementInput<M>& measurement, 
        const map<A, double>& costs, 
        double time = 0.0
    ) = 0;
};