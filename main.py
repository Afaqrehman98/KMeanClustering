import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow

import dbscan
from KMeans import KMeans


def __main__():
    val = input("Which algorithm you want to run? Enter DB for DB Scan and KM for Kmeans ")

    if val == "DB" or val == "db":
        # Load an image:
        image_path = 'tiger_image.png'

        image = Image.open(image_path)
        image = image.resize((25, 25))
        pixels = image.load()

        # getting dimensions of image
        width, height = image.size
        # Turn image into list of vectors (1 vector / pixel):
        vector_list = []

        # using nested loops to iterate through the pixels of an image with a specified width and height. For each
        # pixel, it is extracting the red, green, and blue values (indicated by the [0], [1], and [2] in the
        # current_point variable) and creating a new vector with these values. It then appends this vector to a list
        # called "vector_list"
        for x in range(width):
            for y in range(height):
                current_point = [pixels[x, y][0], pixels[x, y][1], pixels[x, y][2]]

                current_vector = np.array(current_point)
                vector_list.append(current_vector)

        print('Image file with dimensions {}x{} pixels turned into {} vectors.'.format(width, height, len(vector_list)))
        dbscan_clusters = dbscan.db_scan_algorithm(vector_list, 4, 3)

        def clusters_to_image(cluster_per_point_list: list, points: list, width, height):
            assert (len(cluster_per_point_list) == len(points))

            cluster_count = max(cluster_per_point_list) + 1
            inverted_clusters = [[] for _ in range(cluster_count)]

            for i in range(len(cluster_per_point_list)):
                inverted_clusters[cluster_per_point_list[i]].append(points[i])

            mean_colors = [np.array([0, 0, 0]) for _ in range(cluster_count)]
            counter = [0 for _ in range(cluster_count)]
            for i in range(cluster_count):
                for elem in inverted_clusters[i]:
                    mean_colors[i] = np.add(mean_colors[i], elem)
                    counter[i] += 1
                    mean_colors[i] = np.divide(mean_colors[i], np.array([counter[i], counter[i], counter[i]]))

            clustered_image = Image.new('RGB', (width, height))
            pix = clustered_image.load()
            for x in range(width):
                for y in range(height):
                    cl_id = cluster_per_point_list[y + x * height]
                    if cl_id == -1:
                        pix[x, y] = (0, 0, 0)
                    else:
                        curr_pixel = [int(x) for x in mean_colors[cl_id]]
                        pix[x, y] = tuple(curr_pixel)

            return clustered_image

        clustered_image = clusters_to_image(dbscan_clusters, vector_list, width, height)

        # Display the clustered image:
        imshow(np.asarray(clustered_image))
        image_arr = np.asarray(clustered_image)
        image_arr_cp = np.uint8(image_arr)
        image_obj = Image.fromarray(image_arr_cp)
        image_obj.save('output.png')
    else:

        # Reading the image and then changing its color to RGB as open cv using BGR so for image segmentation we
        # convert it into RGB
        # then reshaping the image from 3d array to 2d array for further processing
        # then converting the pixel values from int to float
        image = cv2.imread("tiger_image_min.png")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        k = KMeans(3, 10)
        y_prediction = k.predict(pixel_values)

        # label corresponding to its closest
        # cluster center. The line centers[labels] is using these labels to index into the centers array, effectively
        # replacing each pixel in the original image with its corresponding cluster center. This process is also
        # known as "re-coloring" the image using the cluster centers. The final result is a segmented image where
        # each segment is represented by one of the k cluster centers.
        centers = np.uint8(k.cent())
        y_prediction = y_prediction.astype(int)
        np.unique(y_prediction)
        labels = y_prediction.flatten()
        segmented_image = centers[labels]
        segmented_image = segmented_image.reshape(image.shape)
        plt.figure(figsize=(6, 6))
        plt.imshow(segmented_image)
        plt.show()

        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("C:\\Users\\afaqr\\OneDrive\\Desktop\\output.jpg", segmented_image)


__main__()
