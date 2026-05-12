# %%
import pandas as pd

from simpletransformers.classification import ClassificationModel

# %%
# Training Data

train_data = [
    ["I love this movie", 1],
    ["This film was amazing", 1],
    ["I hate this movie", 0],
    ["Worst movie ever", 0]
]

train_df = pd.DataFrame(train_data)

train_df.columns = ["text", "labels"]

# %%
# Create Model

model = ClassificationModel(
    model_type='bert',
    model_name='bert-base-uncased',
    num_labels=2,
    use_cuda=False
)

# %%
# Train Model

model.train_model(train_df)

# %%
# Predictions

predictions, raw_outputs = model.predict([
    "This movie is fantastic",
    "Terrible acting and bad story"
])

# %%
print(predictions)