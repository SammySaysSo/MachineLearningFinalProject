import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt

image_dir = 'MachineLearningFinalProject/images'
csv_file = 'MachineLearningFinalProject/Data.csv'

# Load the CSV file
df = pd.read_csv(csv_file)

def load_image(image_name, image_class):
    image_path = os.path.join(image_dir, image_class, image_name)
    if os.path.exists(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image
    
sample_row = df.iloc[0]
image_name = sample_row['image_name']
image_class = sample_row['class']

image = load_image(image_name, image_class)

plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
plt.savefig("./test.png")
