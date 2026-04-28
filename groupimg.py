import os
import shutil
import glob
import math
import argparse
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count

# Allow processing of large images without a size limit
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore')

class KMeans:
    def __init__(self, k=3, size=False, resample=32, distance_metric='euclidean'):
        """
        Initialize the KMeans clustering algorithm.

        :param k: Number of clusters.
        :param size: Whether to include image dimensions in the feature vector.
        :param resample: Size to which images will be resampled.
        :param distance_metric: Distance metric to use ('manhattan', 'euclidean', 'cosine', 'chebyshev', 'minkowski').
        """
        self.k = k
        self.cluster = []  # List to store cluster assignments for each image
        self.data = []     # List to store feature vectors of images
        self.end = []      # List to store file paths of images
        self.i = 0         # Counter for assigning initial clusters
        self.size = size   # Flag to include image dimensions in features
        self.resample = resample  # Size to which images will be resampled
        self.distance_metric = distance_metric  # Distance metric to use

    def manhattan_distance(self, x1, x2):
        """
        Calculate the Manhattan distance between two points.

        :param x1: First point.
        :param x2: Second point.
        :return: Manhattan distance.
        """
        return sum(abs(float(x1[i]) - float(x2[i])) for i in range(len(x1)))

    def euclidean_distance(self, x1, x2):
        """
        Calculate the Euclidean distance between two points.

        :param x1: First point.
        :param x2: Second point.
        :return: Euclidean distance.
        """
        return math.sqrt(sum((float(x1[i]) - float(x2[i])) ** 2 for i in range(len(x1))))

    def cosine_distance(self, x1, x2):
        """
        Calculate the Cosine distance between two points.

        :param x1: First point.
        :param x2: Second point.
        :return: Cosine distance.
        """
        dot_product = sum(float(x1[i]) * float(x2[i]) for i in range(len(x1)))
        norm_x1 = math.sqrt(sum(float(x1[i]) ** 2 for i in range(len(x1))))
        norm_x2 = math.sqrt(sum(float(x2[i]) ** 2 for i in range(len(x2))))
        if norm_x1 == 0 or norm_x2 == 0:
            return float('inf')
        return 1 - (dot_product / (norm_x1 * norm_x2))

    def chebyshev_distance(self, x1, x2):
        """
        Calculate the Chebyshev distance between two points.

        :param x1: First point.
        :param x2: Second point.
        :return: Chebyshev distance.
        """
        return max(abs(float(x1[i]) - float(x2[i])) for i in range(len(x1)))

    def minkowski_distance(self, x1, x2, p=3):
        """
        Calculate the Minkowski distance between two points.

        :param x1: First point.
        :param x2: Second point.
        :param p: Order of the norm.
        :return: Minkowski distance.
        """
        return math.pow(sum(abs(float(x1[i]) - float(x2[i])) ** p for i in range(len(x1))), 1/p)

    def calculate_distance(self, x1, x2):
        """
        Calculate the distance between two points based on the specified metric.

        :param x1: First point.
        :param x2: Second point.
        :return: Distance.
        """
        if self.distance_metric == 'manhattan':
            return self.manhattan_distance(x1, x2)
        elif self.distance_metric == 'euclidean':
            return self.euclidean_distance(x1, x2)
        elif self.distance_metric == 'cosine':
            return self.cosine_distance(x1, x2)
        elif self.distance_metric == 'chebyshev':
            return self.chebyshev_distance(x1, x2)
        elif self.distance_metric == 'minkowski':
            return self.minkowski_distance(x1, x2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def read_image(self, im):
        """
        Read an image and extract its features.

        :param im: File path of the image.
        :return: List containing cluster index, feature vector, and file path.
        """
        if self.i >= self.k:
            self.i = 0
        try:
            img = Image.open(im)
            osize = img.size
            img.thumbnail((self.resample, self.resample))
            # Create a feature vector based on histogram of pixel values
            v = [float(p) / float(img.size[0] * img.size[1]) * 100 for p in np.histogram(np.asarray(img))[0]]
            if self.size:
                v += [osize[0], osize[1]]  # Include image dimensions if size flag is set
            pbar.update(1)  # Update the progress bar
            i = self.i
            self.i += 1
            return [i, v, im]
        except Exception as e:
            print(f"Error reading {im}: {e}")
            return [None, None, None]

    def generate_k_means(self):
        """
        Generate new cluster means based on current assignments.

        :return: List of new cluster means.
        """
        final_mean = []
        for c in range(self.k):
            partial_mean = []
            for i in range(len(self.data[0])):
                s = 0.0
                t = 0
                for j in range(len(self.data)):
                    if self.cluster[j] == c:
                        s += self.data[j][i]
                        t += 1
                if t != 0:
                    partial_mean.append(float(s) / float(t))
                else:
                    partial_mean.append(float('inf'))
            final_mean.append(partial_mean)
        return final_mean

    def generate_k_clusters(self, folder):
        """
        Generate k clusters using a thread pool.

        :param folder: Path to the folder containing images.
        """
        pool = ThreadPool(cpu_count())
        result = pool.map(self.read_image, folder)
        pool.close()
        pool.join()
        self.cluster = [r[0] for r in result if r[0] is not None]
        self.data = [r[1] for r in result if r[1] is not None]
        self.end = [r[2] for r in result if r[2] is not None]

    def rearrange_clusters(self):
        """
        Rearrange clusters based on the closest mean.
        """
        is_over = False
        while not is_over:
            is_over = True
            means = self.generate_k_means()
            for x in range(len(self.cluster)):
                distances = [self.calculate_distance(self.data[x], m) for m in means]
                min_dist_index = distances.index(min(distances))
                if self.cluster[x] != min_dist_index:
                    self.cluster[x] = min_dist_index
                    is_over = False

def main():
    """
    Main function to parse arguments, process images, and cluster them.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=True, help="path to image folder")
    ap.add_argument("-k", "--kmeans", type=int, default=15, help="how many groups")
    ap.add_argument("-r", "--resample", type=int, default=128, help="size to resample the image by")
    ap.add_argument("-s", "--size", action="store_true", help="use size to compare images")
    ap.add_argument("-m", "--move", action="store_true", help="move instead of copy")
    ap.add_argument("-d", "--distance", type=str, default='euclidean', choices=['manhattan', 'euclidean', 'cosine', 'chebyshev', 'minkowski'], help="distance metric to use")
    args = vars(ap.parse_args())

    types = ('*.jpg', '*.JPG', '*.png', '*.jpeg')
    image_paths = []
    folder = args["folder"]
    if not folder.endswith("/"):
        folder += "/"
    for file_type in types:
        image_paths.extend(sorted(glob.glob(folder + file_type)))

    n_images = len(image_paths)
    n_folders = int(math.log(args["kmeans"], 10)) + 1

    if n_images <= 0:
        print("No images found!")
        exit()
    if args["resample"] < 16 or args["resample"] > 256:
        print("-r should be a value between 16 and 256")
        exit()

    global pbar
    pbar = tqdm(total=n_images)
    k_means = KMeans(args["kmeans"], args["size"], args["resample"], distance_metric=args["distance"])
    k_means.generate_k_clusters(image_paths)
    k_means.rearrange_clusters()

    for i in range(k_means.k):
        try:
            os.makedirs(folder + str(i + 1).zfill(n_folders))
        except FileExistsError:
            print("Folder already exists")

    action = shutil.move if args["move"] else shutil.copy
    for i in range(len(k_means.cluster)):
        action(k_means.end[i], folder + str(k_means.cluster[i] + 1).zfill(n_folders) + '/' + os.path.basename(k_means.end[i]))

if __name__ == "__main__":
    main()
