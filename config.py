import random

batch_size=64
train_split_pct=0.7
validation_split_pct=0.15
test_split_pct=0.15
epoch_number=100
lr=1e-3
min_delta=1e-4
resize_width=512
resize_height=512

validation_multilabel_threshold=0.4
testing_multilabel_threshold=0.4
testing_binary_threshold=0.4

random.seed(25)