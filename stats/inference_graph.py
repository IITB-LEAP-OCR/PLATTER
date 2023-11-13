import numpy as np
import matplotlib.pyplot as plt
import json


with open('inference_time.json') as json_data:
    data = json.load(json_data)

data_size = [17574, 16490, 17947, 12869, 15730, 19635, 16851, 16184, 17898, 15517]

custom_order = ["CRNN", "MOBILENET", "SAR", "MASTER", "VITSTR", "PARSEQ"]


for key, value in data.items():
    data[key] = [(x / data_size[i])*1000 for i, x in enumerate(value)]
    
print(data)

# Calculate the average for each key
averages = {key: np.mean(value) for key, value in data.items()}


# languages = ['bengali', 'gujarati', 'gurumukhi', 'hindi', 'kannada', 'malayalam', 'odia', 'tamil', 'telugu', 'urdu']

# for lang in range(len(languages)):
#     plt.figure()
#     # Graph for each language
#     labels, values = zip(*[(key, data[key][lang]) for key in custom_order])
#     plt.bar(labels, values, color=['red', 'green', 'blue', 'cyan', 'yellow', 'orange'])
#     plt.xlabel('Models')
#     plt.ylabel('Time in milli seconds')
#     plt.title('Average Inference time per word on ' + languages[lang])
#     # plt.xticks(rotation=20, ha='right')
#     plt.savefig(f'{languages[lang]}.png')
    

# Create a bar plot with custom order
labels, values = zip(*[(key, averages[key]) for key in custom_order])

# print(labels)

# sums = {key: np.mean(value) for key, value in data.items()}
# # print(sums)

# # Create a bar plot
# labels, values = zip(*averages.items())
plt.bar(labels, values, color=['green', 'blue', 'red', 'cyan', 'yellow', 'orange'])
plt.xlabel('Models')
plt.ylabel('Time in milli seconds')
plt.title('Average Inference time per word on various languages')
# plt.xticks(rotation=20, ha='right')
plt.savefig('inference_time2.png')