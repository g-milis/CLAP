import matplotlib.pyplot as plt


layer = list(range(12))

r1 = [6.95, 7.98, 6.79, 6.75, 6.61, 7.16, 6.65, 7.49, 7.41, 6.50, 7.45, 7.30]
r5 = [14.11, 15.04, 14.75, 13.78, 13.93, 14.75, 13.60, 14.15, 14.48, 13.62, 15.34, 14.93]
r10 = [17.33, 17.74, 17.58, 17.48, 16.72, 18.13, 16.80, 17.37, 18.15, 17.35, 19.36, 18.44]


plt.figure()  # Create a new figure with specific size
plt.plot(layer, r1, label='R@1', color='blue')
plt.axhline(y=7.22, color='blue', linestyle='--')
plt.plot(layer, r5, label='R@5', color='orange')
plt.axhline(y=14.01, color='orange', linestyle='--')
plt.plot(layer, r10, label='R@10', color='gray')
plt.axhline(y=16.78, color='gray', linestyle='--')

# Add the custom dummy line to the handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], linestyle='--', color='black'))
labels.append('base')
plt.legend(handles=handles, labels=labels)

plt.title("Impact of initial reweighting layer on retrieval")
plt.xlabel("Initial reweighting layer")
plt.grid(True)
plt.savefig("plots/audiocaps_per_layer.png", dpi=200, bbox_inches="tight")



r1 = [6.95, 7.98, 6.79, 6.75, 6.61, 7.16, 6.65, 7.49, 7.41, 6.50, 7.45, 7.30]
r5 = [14.11, 15.04, 14.75, 13.78, 13.93, 14.75, 13.60, 14.15, 14.48, 13.62, 15.34, 14.93]
r10 = [17.33, 17.74, 17.58, 17.48, 16.72, 18.13, 16.80, 17.37, 18.15, 17.35, 19.36, 18.44]
map10 = [90.75, 90.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


plt.figure()  # Create a new figure with specific size
plt.plot(layer, r1, label='R@1', color='blue')
plt.axhline(y=7.22, color='blue', linestyle='--')
plt.plot(layer, r5, label='R@5', color='orange')
plt.axhline(y=14.01, color='orange', linestyle='--')
plt.plot(layer, r10, label='R@10', color='gray')
plt.axhline(y=16.78, color='gray', linestyle='--')

# Add the custom dummy line to the handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], linestyle='--', color='black'))
labels.append('base')
plt.legend(handles=handles, labels=labels)

plt.title("Impact of initial reweighting layer on zero-shot classification")
plt.xlabel("Initial reweighting layer")
plt.grid(True)
plt.savefig("plots/esc50_per_layer.png", dpi=200, bbox_inches="tight")
