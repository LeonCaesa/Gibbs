#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Tue Jun 30 22:34:19 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""
import matplotlib.pyplot as plt
import numpy as np


def trace_plot(infer_list, var_list):
    """
        Function to do trace plot
        param: infer_list, list of n_chains of dictonary containing mcmc sampling 
        param: var_list, list of string containing variable numbers

    """
    n_chains = len(infer_list)
    warmup = int(len(infer_list[0][var_list[0]]) / 2)
    for j in var_list:
        start = 0
        for sub_chains in range(n_chains):

            plt.plot(range(start, start + warmup + 1),
                     infer_list[sub_chains][j][warmup:])
            start += warmup

            plt.axvline(x=start, color='k', linestyle='--')
            plt.xlabel('Iteration')
            plt.ylabel('Value')
        plt.title('Trace Plot for ' + str(j) + '')
        plt.show()


def get_trace_list(mcmc_list, var_name):
    """
        Function to get trace list for specific var
    """
    warmup = int(len(mcmc_list[0][var_name]) / 2)
    series = np.array([i[var_name][warmup:] for i in mcmc_list])
    return series
