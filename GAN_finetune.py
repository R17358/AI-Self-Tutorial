# Ensure your dataset (e.g., CUB-200-2011) is in the correct format.
# Images should be resized to the same resolution as the pretrained GAN model (e.g., 256x256 for StyleGAN2).

import os
from PIL import Image
from torchvision import transforms

def preprocess_images(input_dir, output_dir, img_size=256):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')  # Ensure RGB format
            img = transform(img)
            save_path = os.path.join(output_dir, img_name)
            img = transforms.ToPILImage()(img)
            img.save(save_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

preprocess_images('path/to/cub200/images', 'path/to/preprocessed/images', img_size=256)

# pip install torch torchvision ninja gdown

import torch
from torchvision.utils import save_image
from torchvision import transforms
from stylegan2_pytorch import ModelLoader  # Install from https://github.com/lucidrains/stylegan2-pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model = ModelLoader(
    name='ffhq',
    network_pkl='pretrained_model_path.pkl',  # Replace with your pretrained model
    truncation=0.7,
    device=device
)


from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

dataset = ImageFolder('path/to/preprocessed/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)


from torch.optim import Adam
from torch.nn import BCELoss

# Optimizers for generator and discriminator
optimizer_g = Adam(model.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_d = Adam(model.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

criterion = BCELoss()

# Fine-tune loop
epochs = 10
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)

        # Train Discriminator
        model.discriminator.zero_grad()
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)

        outputs_real = model.discriminator(real_images)
        loss_real = criterion(outputs_real, real_labels)

        fake_images = model.generator(torch.randn(real_images.size(0), 512).to(device))
        outputs_fake = model.discriminator(fake_images.detach())
        loss_fake = criterion(outputs_fake, fake_labels)

        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        model.generator.zero_grad()
        outputs = model.discriminator(fake_images)
        loss_g = criterion(outputs, real_labels)
        loss_g.backward()
        optimizer_g.step()

        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss_D: {loss_d.item()}, Loss_G: {loss_g.item()}")

    # Save generated samples
    with torch.no_grad():
        generated_images = model.generator(torch.randn(16, 512).to(device))
        save_image(generated_images, f'output/sample_epoch_{epoch+1}.png', normalize=True, range=(-1, 1))


torch.save(model.state_dict(), "fine_tuned_stylegan2.pth")


# to test

import torch
from torchvision.utils import save_image
from stylegan2_pytorch import ModelLoader  # Ensure you have StyleGAN2 installed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load fine-tuned model
model_path = "fine_tuned_stylegan2.pth"
model = ModelLoader(name='ffhq', network_pkl=model_path, truncation=0.7, device=device)
model.generator.load_state_dict(torch.load(model_path, map_location=device))
model.generator.eval()


# Generate latent vectors (batch of latent codes)
latent_dim = 512
num_samples = 1  # Number of images to generate
latent_vectors = torch.randn(num_samples, latent_dim).to(device)

# Generate images
with torch.no_grad():
    generated_images = model.generator(latent_vectors)

# Save the generated image
save_image(
    generated_images, 
    "generated_image.png", 
    normalize=True, 
    range=(-1, 1)
)

# Display the image (optional)
import matplotlib.pyplot as plt
import numpy as np

# Convert image tensor to a NumPy array for visualization
generated_image = generated_images[0].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
generated_image = (generated_image + 1) / 2  # Scale to [0, 1]

plt.imshow(np.clip(generated_image, 0, 1))
plt.axis("off")
plt.show()

