import numpy as np
import os

SIGNS = ["hello", "yes", "no", "nothing", "thank you", "please"]

X = []
y = []

for label, sign in enumerate(SIGNS):
    folder = os.path.join("data", sign)
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            sequence = np.load(os.path.join(folder, file))
            X.append(sequence)
            y.append(label)

X = np.array(X)
y = np.array(y)

# expected shape: (num_sequences, 30, 63)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("Preprocessing complete.")