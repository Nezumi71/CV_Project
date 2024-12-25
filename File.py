import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import os

# Гиперпараметры
input_shape = (512, 512)  # Размер входных изображений
batch_size = 16
EPOCHS = 2


# Создание трансформаций для данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(input_shape),
])


class MedicalDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):

        # Загрузка изображений и масок из директорий

        images = []
        masks = []
        for number_of_MRI in os.listdir(images_dir):
            number_of_MRI_path = os.path.join(images_dir, number_of_MRI)
            if os.path.isdir(number_of_MRI_path):
                for img_name in os.listdir(number_of_MRI_path):
                    img_path = os.path.join(number_of_MRI_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)


        for number_of_MRI in os.listdir(masks_dir):
            number_of_MRI_path = os.path.join(masks_dir, number_of_MRI)
            if os.path.isdir(number_of_MRI_path):
                for mask_name in os.listdir(number_of_MRI_path):
                    mask_path = os.path.join(number_of_MRI_path, mask_name)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        masks.append(mask)


        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)  # Количество изображений

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            image = self.transform(image)  # Примените трансформации к изображению
            mask = self.transform(mask)

        return image, mask




# Загрузка и предобработка данных


train_dataset = MedicalDataset(r'Good\imagesTr', r'Good\LabelsTr', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# print(train_dataset[0][0].shape)  # размер первого изображения

image_tensor = train_dataset[64][0]  # Случайное изображение
# Преобразование тензора PyTorch в NumPy массив
image_np = image_tensor.numpy()
# Изменение порядка осей для OpenCV (из (C, H, W) в (H, W, C))
image_np = np.transpose(image_np, (1, 2, 0))  # Теперь это (256, 256, 3)
cv2.imshow('Image', image_np)
#cv2.waitKey(0)

image_tensor = train_dataset[64][1]  # Случайное изображение
# Преобразование тензора PyTorch в NumPy массив
image_np = image_tensor.numpy()
# Изменение порядка осей для OpenCV (из (C, H, W) в (H, W, C))
image_np = np.transpose(image_np, (1, 2, 0))  # Теперь это (256, 256, 3)
cv2.imshow('Mask', image_np)
#cv2.waitKey(0)

##############################################################################


# Определение нейронной сети
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        out = self.act(x)
        return out


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        out_doble_conv = self.double_conv(x)
        out_down = self.down(out_doble_conv)

        return out_doble_conv, out_down


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, (2, 2), stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        out = self.double_conv(x)

        return out


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        self.out = nn.Conv2d(64, num_classes, (1, 1))

    def forward(self, x):
        sk1, x = self.down1(x)
        sk2, x = self.down2(x)
        sk3, x = self.down3(x)
        sk4, x = self.down4(x)

        x = self.bottleneck(x)

        x = self.up1(x, sk4)
        x = self.up2(x, sk3)
        x = self.up3(x, sk2)
        x = self.up4(x, sk1)

        out = self.out(x)

        return out

device = 'cuda' if torch.cuda.is_available() else 'cpu'     # Переход на графический процессор для ускорения вычислений

# model = UNet().to(device)
# #print(model)
#
#
# input = torch.rand([2, 1, 512, 512]).to(device)        # проверка правильности построения нейросети
# pred = model(input)
# print(pred.shape)


# model = UNet().to(device)
model = torch.load("Unet_save.pt").to(device)

# Оптимизатор и функция потерь
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()



# Обучение сети
for epoch in range(EPOCHS):  # Повторите цикл по набору данных несколько раз
    model.train()
    running_train_loss = []
    true_answer = 0
    train_loop = tqdm(train_loader, leave=False)
    for image, mask in train_loop:
        image = image.to(device)

        mask = mask.float().to(device)
        #labels = torch.eye(10)[labels].to(device)  # Преобразование классов в one-hot вектора

        # Прямой проход + расчёт ошибки
        pred = model(image)
        loss = criterion(pred, mask)

        # Обратный проход (метод градиентного спуска)
        optimizer.zero_grad()  # Обнуление градиентов
        loss.backward()  # Вычисление новых градиентов

        # Шаг оптимизации (обновление параметров модели посредством выбранного метода градиентного спуска)
        optimizer.step()

        # running_train_loss.append(loss.item())
        # mean_train_loss = sum(running_train_loss) / len(running_train_loss)
        #
        # true_answer += (pred.argmax(dim=1) == labels.argmax(dim=1)).sum().item()

        train_loop.set_description(f"Epoch[{epoch + 1}/{EPOCHS}]")      #, train_loss={mean_train_loss}

    # Расчёт значения метрики
    #running_train_accuracy = true_answer / len(trainset)

print('Обучение закончено')



#Визуализация результатов после обучения (пример для тестового изображения)


img_path = r"Good/imagesTs/img13/image.0064.jpg"
img = cv2.imread(img_path)
plt.imshow(img)
plt.show()

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = transform(img)
print("Размер тензора:", img.unsqueeze(0).shape)    # Должно быть (1,1,512,512)

# Предсказание класса для загруженного изображения

# torch.save(model, "Unet_save.pt")
# model = torch.load("Unet_save.pt")

model.eval()
with torch.no_grad():
    test_image = img
    test_mask_predicted = model(test_image.unsqueeze(0))  # Добавьте размерность батча

print(f"Размер предсказанного класса: {test_mask_predicted.shape}")

plt.imshow(test_mask_predicted[0][0])
plt.savefig("Mask_tr.jpg")
plt.show()




