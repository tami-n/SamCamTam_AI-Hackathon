import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import json
import os

json_file = "enhanced_tokengains_results.json"

if not os.path.exists(json_file):
    print(f"No results file found at {json_file}, run tokengains.py first.")
    exit()

## load json
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data.get("detailed_results", [])

#convert to df
records = []
for idx, entry in enumerate(results, start=1):
    records.append({
        "Test #": idx,
        "Strategy": entry.get("strategy"),
        "Original Tokens": entry.get("original_tokens"),
        "Optimized Tokens": entry.get("reduced_tokens"),
        "Token Reduction %": round(entry.get("token_reduction_percent", 0), 2),
        "Original Cost": None,  # Not in detailed_results, can be added if needed
        "Optimized Cost": None,  # Not in detailed_results
        "Cost Reduction %": round(entry.get("cost_savings_percent", 0), 2),
        "Quality Score": round(entry.get("quality_score", 0), 3),
        "Latency (ms)": round(entry.get("latency_ms", 0), 2)
    })

df = pd.DataFrame(records)

# aggregate stats by strat
agg_df = df.groupby("Strategy").agg(
    Avg_Original_Tokens=("Original Tokens", "mean"),
    Avg_Optimized_Tokens=("Optimized Tokens", "mean"),
    Avg_Token_Reduction_Percent=("Token Reduction %", "mean"),
    Avg_Cost_Reduction_Percent=("Cost Reduction %", "mean"),
    Avg_Quality_Score=("Quality Score", "mean"),
    Avg_Latency=("Latency (ms)", "mean"),
    Tests_Count=("Test #", "count")
).reset_index()

# highlight best strat based on weighted score
agg_df["Best Strategy?"] = ""
best_index = (agg_df["Avg_Token_Reduction_Percent"]*0.7 + agg_df["Avg_Quality_Score"]*30).idxmax()
agg_df.loc[best_index, "Best Strategy?"] = "Yes"

# print table
print("\nAggregated Token Gains & Quality Table (by Strategy):")
print(tabulate(agg_df, headers="keys", tablefmt="grid", showindex=False))

# token reduction % vs. quality score
plt.figure(figsize=(10,6))
plt.scatter(agg_df["Avg_Token_Reduction_Percent"], agg_df["Avg_Quality_Score"],
            s=150, c=['gold' if x=="Yes" else 'skyblue' for x in agg_df["Best Strategy?"]])
for i, row in agg_df.iterrows():
    plt.text(row["Avg_Token_Reduction_Percent"]+0.3, row["Avg_Quality_Score"]+0.003, 
             f"{row['Strategy']}{' ★' if row['Best Strategy?']=='Yes' else ''}", fontsize=9)
plt.xlabel("Average Token Reduction %")
plt.ylabel("Average Quality Score")
plt.title("Strategy Performance: Token Reduction vs Quality")
plt.grid(alpha=0.3)
plt.show()

# orginal vs. optimised tokens per test
plt.figure(figsize=(10,6))
bar_width = 0.15
strategies = df["Strategy"].unique()
x = df["Test #"].unique()

for i, strategy in enumerate(strategies):
    strat_df = df[df["Strategy"] == strategy]
    offset = (i - len(strategies)/2) * bar_width
    color_orig = "gold" if strategy == agg_df.loc[best_index, "Strategy"] else "salmon"
    color_opt = "goldenrod" if strategy == agg_df.loc[best_index, "Strategy"] else "lightgreen"
    plt.bar(strat_df["Test #"] + offset, strat_df["Original Tokens"], width=bar_width, label=f"Original ({strategy})", color=color_orig)
    plt.bar(strat_df["Test #"] + offset, strat_df["Optimized Tokens"], width=bar_width, bottom=strat_df["Original Tokens"], label=f"Optimized ({strategy})", color=color_opt)

plt.xlabel("Test #")
plt.ylabel("Tokens")
plt.title("Original vs Optimized Tokens by Strategy (Best Highlighted)")
plt.xticks(x)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()