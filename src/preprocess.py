import numpy as np
import os

SIGNS = ["hello", "yes", "no", "nothing", "thank you", "please", "eat", "drink", "water", "more", "apple", "mother", "father", "book", "walk", "cold", "hot", "black", "carrot", "go", "day", "break", "cow", "monkey", "draw", "type"]
X = []
y = []

clean_counts = {}
messy_counts = {}

for label, sign in enumerate(SIGNS):
    folder = os.path.join("data", sign)
    clean = sorted([f for f in os.listdir(folder) if f.startswith("seq_clean") and f.endswith(".npy")])
    messy = sorted([f for f in os.listdir(folder) if f.startswith("seq_messy") and f.endswith(".npy")])

    clean_counts[sign] = len(clean)
    messy_counts[sign] = len(messy)

    for file in clean + messy:
        sequence = np.load(os.path.join(folder, file))
        X.append(sequence)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print()
print(f"{'Sign':<12} {'Clean':>6} {'Messy':>6} {'Total':>6}")
print("-" * 32)
for sign in SIGNS:
    total = clean_counts[sign] + messy_counts[sign]
    print(f"{sign:<12} {clean_counts[sign]:>6} {messy_counts[sign]:>6} {total:>6}")

np.save("data/X.npy", X)
np.save("data/y.npy", y)

print()
print("Preprocessing complete.")