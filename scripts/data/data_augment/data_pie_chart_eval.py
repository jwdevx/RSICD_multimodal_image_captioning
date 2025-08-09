#!/usr/bin/env python3

import matplotlib.pyplot as plt

# Data from your description
labels = ['5 Sentences', '4 Sentences', '3 Sentences', '2 Sentences', '1 Sentence']
sizes = [724, 1495, 2182, 1667, 4853]

# Optional: explode the largest slice for emphasis
explode = [0, 0, 0, 0, 0.1]  # explode the '1 Sentence' slice

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, explode=explode)
plt.title('Distribution of Sentence Descriptions in RSICD Dataset')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
