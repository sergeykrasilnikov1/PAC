import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch

transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5), (0.5))
                                            ])

# Downloading the MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./MNIST/train", train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True)

test_dataset = torchvision.datasets.MNIST(
    root="./MNIST/test", train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True)


# Printing 25 random images from the training dataset
random_samples = np.random.randint(1, len(train_dataset), (25))

for idx in range(random_samples.shape[0]):
    plt.subplot(5, 5, idx + 1)
    plt.imshow(train_dataset[idx][0][0].numpy(), cmap='gray')
    plt.title(train_dataset[idx][1])
    plt.axis('off')




test1 = train_dataset[0]

def encode_label(j):
    # 5 -> [[0], [0], [0], [0], [0], [1], [0], [0], [0], [0]]
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def shape_data(data):
    features = [np.reshape(x[0][0].numpy(), (784,1)) for x in data]
    #print('features\n', len(features[0]))
    labels = [encode_label(y[1]) for y in data]
    #print('labels\n', len(labels[0]))
    return zip(features, labels)

test2 = [test1]
reshape = shape_data(test2)
train = shape_data(train_dataset)
test = shape_data(test_dataset)

train = list(train)
test = list(test)
print('train', len(train))
print('test', len(test))

def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)

avg_digits = [average_digit(train, i) for i in range(10)]
W = [np.transpose(i) for i in avg_digits]
images = [np.reshape(i, (28, 28)) for i in avg_digits]
for i in images:
    plt.imshow(i)
    plt.show()

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def predict(x, W,b):
    return [sigmoid(np.dot(W[i], x) - b)[0][0] for i in range(len(W))]

b = 80
correct_num = 0
n = len(train)
n = 5000
prob_vectors = []
pred_labels = []
# import tqdm
for idx in range(n):
    prob = predict(train[idx][0], W,b)
    digit = np.where((train[idx][1]) == 1)[0][0]
    prob_vectors.append(prob)
    pred_labels.append(np.argmax(prob))
    if(np.argmax(prob) == digit):
      correct_num += 1
print("Accuracy is", (correct_num / n))

X = []
y = []
number = np.zeros((10,1))

testX = my_scale(av_dig, testX)
for i in range(len(testX)):
    X.append(testX[i])
    y.append(np.argmax(testX[i]))
X = np.array(X)


tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, alpha=0.6)
plt.colorbar(scatter, ticks=range(10), label="Digits")
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE 1st")
plt.ylabel("t-SNE 2nd")
plt.show()

tsne = TSNE(n_components=2, random_state=42)

print(prob_vectors)
embedded_sne = tsne.fit_transform(np.array(prob_vectors))
plt.scatter(embedded_sne[:, 0], embedded_sne[:, 1], c=np.array(pred_labels), cmap="tab10")
plt.colorbar()
plt.title("Res for embeddings")
plt.show()
