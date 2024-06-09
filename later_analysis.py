import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

# cut_layer_results = pd.read_csv("cut_layer_results.csv")
# cut_layer_results["Number of cutted blocks"] = 32 - cut_layer_results["layer"]
# fig, ax1 = plt.subplots(figsize=(12, 6))
# ax2 = ax1.twinx()
# sns.lineplot(data=cut_layer_results, x="Number of cutted blocks", y="accuracy", ax=ax1, label="Accuracy")
# sns.lineplot(data=cut_layer_results, x="Number of cutted blocks", y="prob_avg", ax=ax2, label="Probability of correct answer")
# ax1.set_ylabel("Accuracy")
# ax2.set_ylabel("Probability of correct answer")
# plt.xlabel("Number of cutted blocks")
# plt.title("Removing GD steps Experiment")
# plt.savefig("cut_layer_metrics.png")

# learning_rate_results = pd.read_csv("learning_rate_results.csv")
# # for "lr_scale" = 1.0, fill in the missing values with the accuracy of the first layer
# new_rows = []
# lr1_accuracy = learning_rate_results[learning_rate_results["lr_scale"] == 1.0]["accuracy"].iloc[0]
# for start_layer_num in range(10, 32, 5):
#     new_rows.append({"start_layer_num": start_layer_num, "lr_scale": 1.0, "accuracy": lr1_accuracy})
# learning_rate_results = pd.concat([learning_rate_results, pd.DataFrame(new_rows)], ignore_index=True)
# # create a heatmap
# fig, ax = plt.subplots(figsize=(12, 6))
# heatmap_data = learning_rate_results.pivot(columns="start_layer_num", index="lr_scale", values="accuracy")
# sns.heatmap(heatmap_data, ax=ax, cmap="viridis", annot=True, fmt=".3f")
# plt.title("Scaling Learning Rate Experiment")
# plt.savefig("learning_rate_metrics.png")

results_df = pd.read_csv("skip_blocks_results.csv")
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=results_df, x="k", y="avg_accuracy", ax=ax, label="Model with removed blocks")
plt.axhline(y=0.84, color="red", linestyle="--", label="Original model")
plt.xlabel("Number of blocks removed")
plt.ylabel("Accuracy")
plt.title("Randomly Removing Blocks from [15, 31] Experiment")
plt.grid(True)
plt.legend()
plt.savefig("skip_blocks_metrics.png")
