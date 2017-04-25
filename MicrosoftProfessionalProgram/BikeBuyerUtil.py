# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:32:32 2017

@author: Rita Philavanh
"""
import matplotlib.pyplot as plt

def correlation_matrix(df, colList, outName):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(df[colList].corr(), interpolation='nearest')
    fig.colorbar(cax)
    ax.set_xticklabels(['']+colList)
    ax.set_yticklabels(['']+colList)

    plt.show()
    fig.savefig(outName+'.png', bbox_inches='tight')
          
