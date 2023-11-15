import matplotlib.pyplot as plt

models = ['CRNN_VGG', 'MOBILENET', 'SAR',  'MASTER']


crr_end = [97, 74.59, 92.3, 96.59]

wrr_end = [76.01, 27.66, 55.59, 77.17]

wrr_iso = [87.24, 27.04, 88.91, 89.18]

crr_iso = [98.38, 85.66, 99.14, 98.71]

latency = [179, 154, 513, 183]
latency = [(x*1000)/19635 for x in latency]


# plt.figure(figsize=(10, 6))
# plt.figure(figsize=(8, 4))

plt.plot(models, crr_end, label='CRR End-to-end', marker='o')
plt.plot(models, wrr_end, label='WRR End-to-end', marker='s')
plt.plot(models, wrr_iso, label='WRR Isolated', marker='^')
plt.plot(models, crr_iso, label='CRR Isolated', marker='x')

# Adding labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Accuracies for Different Models')
plt.legend()

# Display the plot
plt.savefig('accuracy.png')

plt.figure()
plt.bar(models, latency, color=['blue'])
plt.xlabel('Models')
plt.ylabel('Time in milli seconds')
plt.title('End-to-end Latency for Different Models')
plt.legend()

# Display the plot
plt.savefig('latency.png')
