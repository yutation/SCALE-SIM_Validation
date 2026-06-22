import pandas as pd
import ast

df = pd.read_csv("merged_verification_results.csv")

def parse_mnk(shape_str):
    shapes = ast.literal_eval(shape_str)
    M, K = shapes[0]
    K2, N = shapes[1]
    return M, K, N

df[["M", "K", "N"]] = df["Input_Shapes"].apply(
    lambda s: pd.Series(parse_mnk(s))
)

df.to_csv("merged_verification_results_mnk.csv", index=False)
print("Done. Output: merged_verification_results_mnk.csv")
print(df[["Input_Shapes", "M", "K", "N"]].head())
