import numpy as np
import collections
import pandas as pd
import os

labels = np.load('./brain_tumor_dataset/labels.npy')
images = np.load('./brain_tumor_dataset/images.npy',allow_pickle=True)

def balance_distribution():
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
   

balance_distribution()

