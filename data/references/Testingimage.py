import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        np.savetxt("layer_conv1.txt", x.cpu().numpy().flatten())
        x = self.bn1(x)
        np.savetxt("layer_bn1.txt", x.cpu().numpy().flatten())
        x = self.pool(F.relu(x))
        np.savetxt("layer_pool1.txt", x.cpu().numpy().flatten())
        x = self.conv2(x)
        np.savetxt("layer_conv2.txt", x.cpu().numpy().flatten())
        x = self.bn2(x)
        np.savetxt("layer_bn2.txt", x.cpu().numpy().flatten())
        x = self.pool(F.relu(x))
        np.savetxt("layer_pool2.txt", x.cpu().numpy().flatten())
        x = self.conv3(x)
        np.savetxt("layer_conv3.txt", x.cpu().numpy().flatten())
        x = self.bn3(x)
        np.savetxt("layer_bn3.txt", x.cpu().numpy().flatten())
        x = self.pool(F.relu(x))
        np.savetxt("layer_pool3.txt", x.cpu().numpy().flatten())
        x = x.view(-1, 256 * 4 * 4)
        np.savetxt("layer_flatten.txt", x.cpu().numpy().flatten())
        x = self.fc1(x)
        np.savetxt("layer_fc1.txt", x.cpu().numpy().flatten())
        x = F.relu(self.bn_fc1(x))
        np.savetxt("layer_bn_fc1.txt", x.cpu().numpy().flatten())
        x = self.dropout(x)
        np.savetxt("layer_dropout.txt", x.cpu().numpy().flatten())
        x = self.fc2(x)
        np.savetxt("layer_fc2.txt", x.cpu().numpy().flatten())
        return x

def load_image_matrix(file_path):
    matrix = np.loadtxt(file_path)
    matrix = matrix.reshape(32, 32, 3)
    matrix = matrix.transpose(2, 0, 1)
    return matrix

model = CNN()
model.load_state_dict(torch.load('C:/Users/7501/Desktop/MCW_Intern/best_model_final.pth'))
model.eval()


def save_output_to_file(output, file_path, classes):
    # Convert the output tensor to a numpy array
    output_np = output.cpu().numpy().flatten()

    # Find the highest value and its corresponding class
    max_value = np.max(output_np)
    max_index = np.argmax(output_np)
    predicted_class = classes[max_index]

    # Open the file for writing
    with open(file_path, 'w') as file:
        # Save all the output values and corresponding class labels
        for i in range(len(output_np)):
            file.write(f"Value: {output_np[i]}, Class: {classes[i]}\n")

        # Save the highest value and its predicted class
        file.write(f"\nHighest Value: {max_value}\n")
        file.write(f"Predicted Class: {predicted_class}\n")


image_matrix_path = "C:/Users/7501/Downloads/projects/Projects_MCW/data/input/image_matrix_deer.txt"
input_matrix = load_image_matrix(image_matrix_path)
reshaped_matrix = np.expand_dims(input_matrix, axis=0)
reshaped_matrix = torch.tensor(reshaped_matrix, dtype=torch.float32)

print("Reshaped matrix shape: ", reshaped_matrix.shape)

classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Get the model's output
with torch.no_grad():
    output = model(reshaped_matrix)

# Save the output to a text file
save_output_to_file(output, "C:/Users/7501/Downloads/projects/Projects_MCW/report/Output_pytorch.txt", classes)

print("Model output saved to output.txt")

print("Model output:", output)
