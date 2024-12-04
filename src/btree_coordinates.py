import csv
import pickle
import time
from sklearn.metrics import mean_squared_error


class BTreeNode:
    def __init__(self, t):
        self.t = t  # Minimum degree (defines the range for the number of keys)
        self.keys = []  # List of keys (latitude, index)
        self.children = []  # List of child pointers
        self.leaf = True  # True if the node is a leaf

    def insert_non_full(self, key, value):
        """Insert a key into a non-full node."""
        i = len(self.keys) - 1

        # If this is a leaf node
        if self.leaf:
            # Find the location to insert the new key
            while i >= 0 and key < self.keys[i][0]:
                i -= 1
            self.keys.insert(i + 1, (key, value))
        else:
            # Find the child which will have the new key
            while i >= 0 and key < self.keys[i][0]:
                i -= 1
            i += 1

            # Check if the found child is full
            if len(self.children[i].keys) == 2 * self.t - 1:
                self.split_child(i)

                # After split, the middle key of children[i] goes up, so we may need to insert into the next child
                if key > self.keys[i][0]:
                    i += 1

            self.children[i].insert_non_full(key, value)

    def split_child(self, i):
        """Split the i-th child of this node."""
        t = self.t
        y = self.children[i]
        z = BTreeNode(t)
        z.leaf = y.leaf
        z.keys = y.keys[t:]  # The second half of keys goes to the new node
        y.keys = y.keys[:t - 1]  # The first half remains in the old node

        # If not a leaf, move the corresponding child pointers
        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]

        # Insert the middle key into this node
        self.children.insert(i + 1, z)
        self.keys.insert(i, y.keys.pop())

    def search(self, key, tolerance=1e-6):
        """Search for a key in the B-tree."""
        i = 0
        while i < len(self.keys) and key > self.keys[i][0]:
            i += 1

        # If the key is found within the tolerance
        if i < len(self.keys) and abs(key - self.keys[i][0]) < tolerance:
            return self.keys[i][1]

        # If this is a leaf node
        if self.leaf:
            return None

        # Recur to the appropriate child
        return self.children[i].search(key, tolerance)

    def print_tree(self, level=0):
        print("Level", level, "Keys:", [key for key, _ in self.keys])
        if not self.leaf:
            for child in self.children:
                child.print_tree(level + 1)

    def write_tree_to_file(self, file, level=0):
        """Recursively write the tree structure to a text file."""
        indent = "    " * level  # Indentation based on level
        file.write(f"{indent}Level {level} Keys: {[key for key, _ in self.keys]}\n")
        if not self.leaf:
            for child in self.children:
                child.write_tree_to_file(file, level + 1)


class BTree:
    def __init__(self, t):
        self.root = BTreeNode(t)
        self.t = t

    def insert(self, key, value):
        """Insert a key-value pair into the B-tree."""
        root = self.root

        # If the root node is full, split it
        if len(root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(self.t)
            new_root.children.append(root)
            new_root.leaf = False
            new_root.split_child(0)
            self.root = new_root

        self.root.insert_non_full(key, value)

    def search(self, key):
        """Search for a key in the B-tree."""
        return self.root.search(key)

    def save_tree_to_file(self, file_path):
        """Save the entire tree structure to a text file."""
        with open(file_path, 'w') as file:
            self.root.write_tree_to_file(file)

    def save_tree_to_pickle(self, file_path):
        """Save the B-tree to a pickle file."""
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_tree_from_pickle(file_path):
        """Load the B-tree from a pickle file."""
        with open(file_path, 'rb') as file:
            return pickle.load(file)


# Function to read CSV file and load data into the B-tree
def load_csv_to_btree(file_path, btree):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            latitude = float(row[0])
            index = int(row[1])
            if index == 1000000:
                break
            btree.insert(latitude, index)

def test_btree_performance(btree, test_file_path):
    test_latitudes = []
    ground_truth_indices = []

    # Load test data
    with open(test_file_path, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            test_latitudes.append(float(row[0]))  # Latitude
            ground_truth_indices.append(float(row[1]))  # Ground truth index

    # Start testing time
    start_time = time.time()

    predicted_indices = []
    for lat in test_latitudes:
        pred_index = btree.search(lat)
        if pred_index is not None:
            predicted_indices.append(pred_index)
        else:
            # If no prediction found, append NaN
            predicted_indices.append(None)

    # Calculate time taken for testing
    elapsed_time = time.time() - start_time
    print(f"Time taken for testing {len(test_latitudes)} entries: {elapsed_time:.2f} seconds")

    # Calculate Mean Squared Error (MSE)
    valid_predictions = [p for p in predicted_indices if p is not None]
    valid_ground_truth = [gt for p, gt in zip(predicted_indices, ground_truth_indices) if p is not None]

    mse = mean_squared_error(valid_ground_truth, valid_predictions)
    print(f"Mean Squared Error (MSE) of predictions: {mse:.4f}")

    # Print predictions that differ from the ground truth
    for lat, pred, gt in zip(test_latitudes, predicted_indices, ground_truth_indices):
        if pred is not None and pred != gt:
            print(f"Latitude: {lat}, Predicted Index: {pred}, Ground Truth Index: {gt}")


# Example usage
# Step 1: Create a B-tree with a minimum degree t = 2
btree = BTree(t=5000)

# Step 2: Read data from CSV and load into B-tree
csv_file = '../data/sydneyUniqueSortedLatitudes.csv'  # Replace with your CSV file path
load_csv_to_btree(csv_file, btree)

# Save the B-tree to a pickle file
pickle_file = 'btree_structure.pkl'
btree.save_tree_to_pickle(pickle_file)

# Step 3: Load the B-tree from the pickle file
loaded_btree = BTree.load_tree_from_pickle(pickle_file)

# Test the B-tree performance and compute MSE
test_file_path = '../data/testLatitude'  # Replace with your test data file path
test_btree_performance(loaded_btree, test_file_path)
