# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:58:03 2020

@author: afpsaros
"""


import matplotlib.pyplot as plt

with open("apo_results.txt", "r") as f:
    lines = [ line.strip( ) for line in list(f) ]

print(lines)

nus = [float(lines[i][5:8]) for i in range(0, len(lines), 2)]

depths = [int(lines[i].split(',')[0][2:]) for i in range(1, len(lines), 2)]

print(depths)

widths = [lines[i].split(',')[1] for i in range(1, len(lines), 2)]

widths = [int(w.split(')')[0][1:]) for w in widths]

print(widths)

plt.title('nu vs optimal depth')
plt.plot(nus, depths, '-o')
plt.xlabel('nu')
plt.ylabel('depth')
plt.show()

plt.title('nu vs optimal width')
plt.plot(nus, widths, '-o')
plt.xlabel('nu')
plt.ylabel('width')
plt.show()