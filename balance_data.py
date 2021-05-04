import numpy as np
import collections
import pandas as pd
import os
from collections import Counter
import random

#takes in down-scaled and original image matrices and balances
#the distribution of the data based on class labels to ensure
#a balance of the different class of tumors is being used in
#all datasets:
#70% of class 1, 2 and 3 will be used for the training data
#10% of class 1, 2, and 3 will be used for the validation data
#10% of class 1, 2, and 3 will be used for the testing data
def balance_data(x, y):
   #load labels
   labels = np.load('./brain_tumor_dataset/labels.npy')

   #get counts of labels, 80% of each class goes to training, 10% goes to val and testing each
   label_counts = Counter(labels)
   class_1_training_split = int(0.8 * label_counts[1])
   class_2_training_split = int(0.8 * label_counts[2])
   class_3_training_split = int(0.8 * label_counts[3])
   class_1_test_val_split = int(0.1 * label_counts[1]) + class_1_training_split
   class_2_test_val_split = int(0.1 * label_counts[2]) + class_2_training_split
   class_3_test_val_split = int(0.1 * label_counts[3]) + class_3_training_split

   #arrays for each class
   class_1_samples = []
   class_2_samples = []
   class_3_samples = []

   #split up the samples based on classes
   for i in range(len(labels)):
      if labels[i] == 1:
         class_1_samples.append([x[i], y[i]])
      elif labels[i] == 2:
         class_2_samples.append([x[i], y[i]])
      elif labels[i] == 3:
         class_3_samples.append([x[i], y[i]])

   #split up data
   training_data = class_1_samples[:class_1_training_split] + class_2_samples[:class_2_training_split] + class_3_samples[:class_3_training_split]
   valiation_data = class_1_samples[class_1_training_split:class_1_test_val_split] + class_2_samples[class_2_training_split:class_2_test_val_split] + class_3_samples[class_3_training_split:class_3_test_val_split]
   testing_data = class_1_samples[class_1_test_val_split:] + class_2_samples[class_2_test_val_split:] + class_3_samples[class_3_test_val_split:]

   #get x and y splits
   x_training, x_validation, x_testing = [], [], []
   y_training, y_validation, y_testing = [], [], []

   for sample in training_data:
      x_training.append(sample[0])
      y_training.append(sample[1])

   for sample in valiation_data:
      x_validation.append(sample[0])
      y_validation.append(sample[1])

   for sample in testing_data:
      x_testing.append(sample[0])
      y_testing.append(sample[1])

   return np.array(x_training), np.array(x_validation), np.array(x_testing), np.array(y_training), np.array(y_validation), np.array(y_testing)

def balance_distribution():
   labels = np.load('./brain_tumor_dataset/labels.npy')
   images = np.load('./brain_tumor_dataset/images.npy', allow_pickle=True)
   print(collections.Counter(labels))
   data = np.vstack((images,labels))
   
   df = pd.DataFrame(data=data.T, columns=['image', 'label'])
   class1 = df.loc[df['label'] == 1]

   # Randomly select 708
   class2 = df.loc[df['label'] == 2].sample(n=708, random_state=42)
   class3 = df.loc[df['label'] == 3].sample(n=708, random_state=42)

   balanced_df = pd.concat([class1, class2, class3]).sample(frac=1)
   
   if not os.path.exists('balanced_dataset'):
      os.makedirs('balanced_dataset')

   np.save('balanced_dataset/balanced_images.npy', balanced_df['image'].to_numpy())
   np.save('balanced_dataset/balanced_labels.npy', balanced_df['label'].to_numpy())
   


