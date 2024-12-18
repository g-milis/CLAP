import matplotlib.pyplot as plt


layer = list(range(12))

r1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
r5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
r10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


plt.figure(figsize=(8, 5))  # Create a new figure with specific size
plt.plot(layer, r1, label='R@1')
plt.plot(layer, r5, label='R@5')
plt.plot(layer, r10, label='R@10')
plt.title("Impact of initial reweighting layer on retrieval")
plt.xlabel("Initial reweighting layer")
plt.legend()
plt.grid(True)
plt.savefig("retrieval_per_layer.png", dpi=200, bbox_inches="tight")
