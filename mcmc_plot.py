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
import pandas as pd

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


def plot_v_density(v_list):
    """
        Function to demonstrate the density of v inference result, plot are sorted according to the mean v in each chain
    """
    sorted_v_list = v_list.copy()
    n_chains = len(v_list)
    for j in range(len(v_list)):
        v_temp = sorted(v_list[j].T, key=np.mean)

        sorted_v_list[j]= np.array(v_temp).T

    v_table = pd.concat([pd.DataFrame(sorted_v_list[i]) for i in range(n_chains)], axis=0, keys=list(range(n_chains)))
    v_table.plot.density()
    plt.title('V Density Plot')
    plt.xlabel('v range')
    return v_table