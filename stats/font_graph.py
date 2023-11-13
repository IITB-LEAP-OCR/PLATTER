import numpy as np
import matplotlib.pyplot as plt
import json


with open('data_stats.json') as json_data:
    data = json.load(json_data)
    
    
    
data = {
    "32": 32844,
    "33": 33165,
    "34": 32819,
    "35": 32870,
    "36": 32450,
    "37": 32525,
    "38": 32605,
    "39": 32763,
    "40": 32772,
    "41": 32959,
    "42": 32637,
    "43": 32554,
    "44": 32571,
    "45": 32876,
    "46": 32963,
    "47": 32583,
    "48": 32951,
    "49": 32778,
    "50": 32640,
    "51": 33113,
    "52": 32842,
    "53": 32908,
    "54": 32781,
    "55": 33046,
    "56": 32756,
    "57": 32564,
    "58": 32662,
    "59": 32639,
    "60": 32799,
    "61": 32811,
    "62": 32796,
    "63": 32887,
    "64": 32801
}

ranges = ['32-40', '40-50', '50-60', '60-65']


# plot pie chart
labels = ranges
sizes = [sum([data[str(i)] for i in range(32, 40)]), sum([data[str(i)] for i in range(40, 50)]), sum([data[str(i)] for i in range(50, 60)]), sum([data[str(i)] for i in range(60, 65)])]
explode = (0, 0, 0, 0)  # explode 1st slice
colors = ['red', 'green', 'blue', 'cyan']
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')

plt.savefig('data_stats.png')
