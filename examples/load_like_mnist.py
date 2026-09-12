"""Load a vehicular dataset the same way you would load MNIST."""

from torch.utils.data import DataLoader

from cpsbench import SynCAN, list_datasets

print("Registered datasets:")
for row in list_datasets():
    print(f"  {row['name']:16} {row['status']:8} shape={row['input_shape']}")

train = SynCAN(root="./data", split="train", download=True)
test = SynCAN(root="./data", split="test", download=True)

window, label = train[0]
print(f"One sample: window {tuple(window.shape)} label {label} ({train.classes[label]})")
print(f"Train windows: {len(train)}  Test windows: {len(test)}")

loader = DataLoader(train, batch_size=64, shuffle=True)
batch, labels = next(iter(loader))
print(f"Batch: {tuple(batch.shape)} labels {tuple(labels.shape)}")
