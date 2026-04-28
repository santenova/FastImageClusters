
**FastImageClusters: A Fast Python Image Cluster, Unsupervised Image Classification**
=====================================================

**Introduction**
--------------

FastImageClusters is a Python library that enables fast and efficient image clustering using unsupervised learning techniques. It allows you to cluster images based on their features, such as pixel values, and assign them to different groups or clusters.

**Features**
------------

* Supports various distance metrics (Manhattan, Euclidean, Cosine, Chebyshev, Minkowski)
* Includes image dimensions in the feature vector
* Resamples images to a specified size for faster processing
* Uses parallel processing with multiple threads for improved performance
* Generates new cluster means based on current assignments

**Usage**
---------

To use FastImageClusters, simply clone this repository and run the `main.py` script. You can customize the clustering process by passing command-line arguments:

* `-f`: Specify the path to the folder containing images.
* `-k`: Set the number of clusters (default: 15).
* `-r`: Set the size to resample images (default: 128).
* `-s`: Include image dimensions in the feature vector (default: False).
* `-m`: Move files instead of copying them (default: False).
* `-d`: Specify the distance metric to use (default: Euclidean).

Here's an example command:

```

# Clone repo
git clone git@github.com:santenova/FastImageClusters.git;

# cd to repo
cd FastImageClusters;

# Create the virtual environment
python3.9 -m venv venv

# Activate the virtual environment (Unix/MacOS)
source venv/bin/activate

# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Install the required packages
python3.9 -m pip install -r requirements.txt

# Do It
python3.9 groupimg.py -f test-images -k 7 -r 64 -s -m

# Check output
ls  test-images/*

```


**Requirements**
----------------

* Python 3.7 or later
* Pillow library (for reading and processing images)

**License**
---------

FastImageClusters is released under the MIT License. See the `LICENSE` file for more information.

**Contributing**
--------------

If you'd like to contribute to FastImageClusters, please fork this repository and submit a pull request with your changes. Make sure to follow our coding style guidelines and include tests for any new features or bug fixes.

**Acknowledgments**
----------------

FastImageClusters is built on top of the Pillow library and uses parallel processing with multiple threads. We'd like to thank the authors of these libraries for their excellent work.
