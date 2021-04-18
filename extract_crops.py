import numpy as np
import cv2
import matplotlib.pyplot as plt

labels = np.load('./brain_tumor_dataset/labels.npy')
images = np.load('./brain_tumor_dataset/images.npy',allow_pickle=True)
masks = np.load('./brain_tumor_dataset/masks.npy',allow_pickle=True)

images_cropped = []

def get_bounding_box(mask):
    xmin, ymin, xmax, ymax = 0, 0, 0, 0

    for row in range(mask.shape[0]):
        if mask[row, :].max() != 0:
            ymin = row
            break

    for row in range(mask.shape[0] - 1, -1, -1):
        if mask[row, :].max() != 0:
            ymax = row
            break

    for col in range(mask.shape[1]):
        if mask[:, col].max() != 0:
            xmin = col
            break

    for col in range(mask.shape[1] - 1, -1, -1):
        if mask[:, col].max() != 0:
            xmax = col
            break

    return xmin, ymin, xmax, ymax

def crop_to_bbox(image, bbox, crop_margin=10):
    x1, y1, x2, y2 = bbox

    # force a squared image
    max_width_height = np.maximum(y2 - y1, x2 - x1)
    y2 = y1 + max_width_height
    x2 = x1 + max_width_height

    # in case coordinates are out of image boundaries
    y1 = np.maximum(y1 - crop_margin, 0)
    y2 = np.minimum(y2 + crop_margin, image.shape[0])
    x1 = np.maximum(x1 - crop_margin, 0)
    x2 = np.minimum(x2 + crop_margin, image.shape[1])

    return image[y1:y2, x1:x2]

dim_cropped_image = 250
for i in range(images.shape[0]):
    bbox = get_bounding_box(masks[i])
    image = crop_to_bbox(images[i], bbox, 20)
    image = cv2.resize(image, dsize=(dim_cropped_image, dim_cropped_image), interpolation=cv2.INTER_CUBIC)
    plt.imshow(image, cmap='bone')
    plt.imsave("original_cropped_images/" + str(i) + ".png", image, cmap='bone')