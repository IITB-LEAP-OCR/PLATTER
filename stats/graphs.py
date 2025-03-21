import matplotlib.pyplot as plt

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


font_sizes = list(data.keys())
no_of_images = list(data.values())

font_sizes = list(data.keys())
no_of_images = list(data.values())

# Create a histogram
plt.bar(font_sizes, no_of_images, color='blue', edgecolor='black')
plt.xlabel('Font Size')
plt.ylabel('Number of Images')
plt.title('Histogram of Font Size Distribution')
plt.xticks(rotation=45, ha='right')  




plt.savefig('font_size_distribution2.png', bbox_inches='tight')