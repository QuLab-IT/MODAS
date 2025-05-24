import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dims=32):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dims, 4, 2, 1)
        self.conv2 = nn.Conv2d(hidden_dims, hidden_dims, 4, 2, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class Decoder(nn.Module):
    def __init__(self, out_channels=1, hidden_dims=32):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(hidden_dims, hidden_dims, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dims, out_channels, 4, 2, 1)

    def forward(self, x):
        return torch.sigmoid(self.deconv2(F.relu(self.deconv1(x))))


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=64, embedding_dim=32):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, z):
        flat_z = z.view(-1, self.embedding_dim)
        distances = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.T
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = torch.argmin(distances, 1).unsqueeze(1)
        quantized = self.embedding(indices).view(z.shape)
        return quantized.detach() + (quantized - z).detach()


class VQVAEFeatureProcessor:
    def __init__(self, patch_grid=8, overlap_ratio=0.25):
        self.encoder = Encoder().eval()
        self.decoder = Decoder().eval()
        self.quantizer = VectorQuantizer().eval()
        self.grid_size = patch_grid
        self.n_segments = patch_grid ** 2
        self.overlap_ratio = overlap_ratio

    def _segment_csdm(self, csd_matrix):
        h, w = csd_matrix.shape
        patch_h = h // self.grid_size
        patch_w = w // self.grid_size
        step_h = int(patch_h * (1 - self.overlap_ratio))
        step_w = int(patch_w * (1 - self.overlap_ratio))

        patches = []
        for i in range(0, h - patch_h + 1, step_h):
            for j in range(0, w - patch_w + 1, step_w):
                patch = csd_matrix[i:i + patch_h, j:j + patch_w]
                if patch.shape == (patch_h, patch_w):
                    patches.append(patch)
                if len(patches) >= self.n_segments:
                    break
            if len(patches) >= self.n_segments:
                break
        return patches

    def _select_diagonal_patches(self, patches):
        selected = []
        for i in range(7):  # 7 diagonally adjacent
            idx = i * (self.grid_size + 1)
            if idx < len(patches):
                selected.append(patches[idx])
        return selected

    def _process_patch(self, patch):
        patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            z = self.encoder(patch_tensor)
            z_q = self.quantizer(z)
            x_recon = self.decoder(z_q)
        return x_recon.squeeze().numpy()

    def process(self, csdm_flat_vector):
        N = int(np.sqrt(csdm_flat_vector.shape[0]))
        csdm_matrix = csdm_flat_vector.reshape(N, N)

        segments = self._segment_csdm(csdm_matrix)
        selected = self._select_diagonal_patches(segments)
        reconstructed = [self._process_patch(patch) for patch in selected]

        return reconstructed  # list of 2D arrays