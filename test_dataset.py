from datasets import load_dataset
import pandas as pd
import  numpy as np
ds = load_dataset("mteb/stackexchange-clustering")

for d in ds:
    exec("sentences_" + d + " = np.array([])")
    exec("labels_" + d + " = np.array([])")
    for s in ds[d]["sentences"]:
        exec("sentences_" + d + " = np.append(sentences_" + d + ", s)")
    for l in ds[d]["labels"]:
        exec("labels_" + d + " = np.append(labels_" + d + ", l)")

data_test = np.array([labels_test, sentences_test]).T
df_test = pd.DataFrame(data=data_test, columns=["labels", "sentences"])
# =============================================================================
# df_test = df_test.sample(10_000, ignore_index=True)
# =============================================================================
print(np.unique(df_test.labels).shape)
df_test.to_csv("data/stack_exchange_train.csv")

data_validation = np.array([labels_validation, sentences_validation]).T
df_validation = pd.DataFrame(data=data_validation, columns=["labels", "sentences"])
# =============================================================================
# df_validation = df_validation.sample(10_000, ignore_index=True)
# =============================================================================
print(np.unique(df_validation.labels).shape)
df_validation.to_csv("data/stack_exchange_val.csv")