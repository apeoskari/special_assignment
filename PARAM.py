# ------------------ Automation Parameters ------------------

AUTOMATIC = True                # Run with or with minimal user input and validation in between
CLUSTER_COUNT = 250             # How many clusters KMeans should make when deciding categories as a default.
AUTO_CLUSTER_COUNT = 15         # How many clusters to pick to choose words from if if 0, will ask.
AUTO_WORDS_PER_CLUSTER = 5      # How many words to pick from a cluster
AUTO_WORDS_PER_BAND = 4         # how many words to pick from a distance category
CLUSTERING_GRAPHS = False       # To make or not to make clustering graphs


# ------------------ General Parameters ------------------

LANG = "fi"                     # Model language
MIN_WORD_LEN = 4                # Minimum word length
MAX_WORD_LEN = 10               # Maximum word length
TOTAL_WORDS = 15000             # Select top X many words to even look at. If commented away, will look at all
LEMMA_BATCH_SIZE = 2000         # How many neighboring tokens to see while lemmatizing
WORD_TYPE = "NOUN"              # Select the deisred wordtype

PLOT_PATH = f"plots/{LANG}/test"     # Relative path to which to save plots to


# ------------------ Clustering Parameters ------------------

SEEDS = []                      # Possible seed words for cluster centroids. The rest will be random. Example of seeds below.
# SEEDS = ["koivu", "auto", "lapio", "äiti"]
SNAP_SEEDS = True
VIEW_CLUSTERS = 20              # How many clusters you want to check
TOP_WORDS = 15                  # Showcase top X words per cluster
PCA = True                      # To run PCA or not
PCA_COUNT = 250                 # What number of PC's to get.


# ------------------ Target Words ------------------

TARGET_WORDS = []               # Ready target words, example below
# TARGET_WORDS = ["koivu", "auto", "lapio", "äiti"]


# ------------------ Primer Words ------------------

DISTANCES = [(0.0, 0.0025), 
             (0.005, 0.01), 
             (0.10, 1.0)]       # In which relative closeness bands you wish to set the 2nd primer words. 1st primers will be from the first.
