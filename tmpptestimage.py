from sklearn.metrics import accuracy_score
import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df1 = pd.read_csv('MachineLearningFinalProject/Data.csv')
df2 = pd.read_csv('MachineLearningFinalProject/extra_hard_samples.csv')
ds = pd.concat([df1, df2], axis=0).reset_index(drop=True)

y = ds["class"]
images = ds["image_name"]
ds = ds.drop('class', axis=1)
ds = ds.drop('image_name', axis=1)
X=ds

image_dir2 = 'MachineLearningFinalProject/images'
image_dir = 'MachineLearningFinalProject/new_images'
csv_file = 'MachineLearningFinalProject/Data.csv'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
forest = RandomForestClassifier(n_estimators=100,max_depth=10, random_state=42)
forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)

# Calculate test accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Load the CSV file
df11 = pd.read_csv('MachineLearningFinalProject/Data.csv')
df22 = pd.read_csv('MachineLearningFinalProject/extra_hard_samples.csv')
df = pd.concat([df11, df22], axis=0).reset_index(drop=True)
# df = pd.read_csv(csv_file)

def load_image(image_name, image_class):
    image_path = os.path.join(image_dir, image_class, image_name)
    if os.path.exists(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image
    
def load_imageAgain(image_name, image_class):
    image_path = os.path.join(image_dir2, image_class, image_name)
    if os.path.exists(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image

# Predict on the test set
y_pred = forest.predict(X_test)

# Identify correct predictions
correct_indices = np.where(y_pred == y_test)[0]  # Indices of correctly predicted samples
print(f"Number of correct predictions: {len(correct_indices)}")

# Get a subset of X_test and y_test for correct predictions
X_test_correct = X_test.iloc[correct_indices]
y_test_correct = y_test.iloc[correct_indices]

correct_samples = df.iloc[correct_indices]

# Display a 5x5 grid with 5 images from each of 5 classes
grid_size = 5  # Number of images per class
num_classes = 5  # Number of classes to display
images_per_class = grid_size

# Get unique classes from correct samples
unique_classes = correct_samples["class"].unique()

# Select exactly 5 classes (adjust if fewer classes exist)
selected_classes = unique_classes[:num_classes]

# Prepare a list to hold selected images
selected_samples = []

# Collect `images_per_class` images from each class
for cls in selected_classes:
    class_samples = correct_samples[correct_samples["class"] == cls]
    selected_samples.extend(class_samples.head(images_per_class).to_dict("records"))

# Plot the selected images
plt.figure(figsize=(12, 12))  # Adjust figure size for better visualization

for i, sample in enumerate(selected_samples):
    image_name = sample['image_name']
    image_class = sample['class']
    
    # Load the image
    image = load_image(image_name, image_class)
    if image is None:
        image = load_imageAgain(image_name, image_class)
    
    if image is not None:
        plt.subplot(grid_size, grid_size, i + 1)  # Create subplot for each image
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # Turn off axis
        plt.title(f"Class: {image_class}", fontsize=10)  # Add title (class name)
    else:
        print(f"Image {image_name} in class {image_class} not found.")
        continue

    # Stop after filling the grid
    if i + 1 == grid_size * grid_size:
        break

plt.tight_layout()  # Adjust spacing between plots
plt.show()



# # Display a 5x5 grid of correct predictions
# grid_size = 5  # Number of rows and columns
# selected_samples = correct_samples.head(grid_size**2)  # Select the first 25 correct samples

# plt.figure(figsize=(12, 12))  # Adjust figure size for better visualization

# for i, (_, row) in enumerate(selected_samples.iterrows()):
#     image_name = row['image_name']
#     image_class = row['class']
    
#     # Load the image
#     image = load_image(image_name, image_class)
    
#     if image is not None:
#         plt.subplot(grid_size, grid_size, i + 1)  # Create subplot for each image
#         plt.imshow(image, cmap='gray')
#         plt.axis('off')  # Turn off axis
#         plt.title(f"Class: {image_class}", fontsize=10)  # Add title (predicted class)
#     else:
#         print(f"Image {image_name} in class {image_class} not found.")

# plt.tight_layout()  # Adjust spacing between plots
# plt.show()


# for _, row in correct_samples.iterrows():
#     image_name = row['image_name']
#     image_class = row['class']
    
#     # Load the image
#     image = load_image(image_name, image_class)
    
#     # Display the image if available
#     if image is not None:
#         plt.figure(figsize=(4, 4))
#         plt.imshow(image, cmap='gray')
#         plt.axis('off')
#         plt.title(f"Class: {image_class}")
#         plt.show()
#     else:
#         print(f"Image {image_name} in class {image_class} not found.")