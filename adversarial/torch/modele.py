"""Le meme CNN que `src/model.py`, mais en PyTorch.

Architecture identique, a la virgule pres :

    Conv2d(1 -> 32, k=3, pad=1) -> ReLU -> MaxPool2d(2)
    Conv2d(32 -> 64, k=3, pad=1) -> ReLU -> MaxPool2d(2)
    Flatten (3136) -> Linear(3136 -> 128) -> ReLU -> Linear(128 -> 10)

Pourquoi ce fichier existe : la version faite main (`src/`) est la reference
pedagogique, mais elle est limitee par le BLAS du CPU. PyTorch permet de
continuer a entrainer le MEME modele sur GPU, et de grossir l'architecture
sans que le temps d'entrainement explose.

Point cle : les fichiers de poids `.npz` sont interchangeables.
On peut donc :
  - charger dans PyTorch un modele entraine en NumPy (et continuer dessus)
  - sauvegarder depuis PyTorch dans le format NumPy
  - verifier la parite numerique des deux implementations
"""

import re

import numpy as np
import torch
import torch.nn as nn


class CNN(nn.Module):
    """Meme architecture que la version NumPy (voir src/model.py).

    `large=True` double les largeurs : deux fois plus de canaux dans les deux
    convolutions, et 256 neurones au lieu de 128 en fully connected. On passe
    de 421 642 a environ 1,7 million de parametres. Sert a tester l'hypothese
    "plus de capacite = plus de robustesse" (Madry et al. montrent que la
    robustesse augmente avec la taille, mais seulement a attaque d'entrainement
    EGALE et assez forte).

    [warn] La conversion `.npz` avec le moteur fait main ne s'applique qu'a
    l'architecture standard : les poids du modele large n'ont pas de
    correspondance cote NumPy.
    """

    def __init__(self, num_classes=10, large=False):
        super().__init__()
        self.large = large
        c1, c2, f = (64, 128, 256) if large else (32, 64, 128)
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(c2 * 7 * 7, f)
        self.fc2 = nn.Linear(f, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialisation He, identique a `np.random.randn(...) * sqrt(2/fan_in)`."""
        for m in (self.conv1, self.conv2, self.fc1, self.fc2):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# --------------------------------------------------------------------------
#  Conversion avec le format .npz de la version NumPy
# --------------------------------------------------------------------------
#
# Dans `src/model.py`, les cles sont nommees par INDEX DE COUCHE :
#   conv_{i}_kernels / conv_{i}_bias     (i = 0 et 3)
#   dense_{i}_W / dense_{i}_b            (i = 7 et 9)
# L'index correspond a la position de la couche dans la liste du modele NumPy
# (Conv2D, ReLU, MaxPool2D, Conv2D, ReLU, MaxPool2D, Flatten, Dense, ReLU, Dense).
# Ces indices sont donc figes par l'architecture NumPy et ne doivent pas bouger.

def _trier_cles(cles, prefixe, suffixe):
    """Trie conv_0, conv_3... par index croissant."""
    def indice(cle):
        return int(re.search(r"_(\d+)_", cle).group(1))
    return sorted([c for c in cles if c.startswith(prefixe) and c.endswith(suffixe)],
                  key=indice)


def charger_npz(modele, chemin):
    """Charge un fichier .npz (format de la version NumPy) dans le modele PyTorch.

    Entree : (N, 1, 28, 28) dans les deux cas. Aucune transposition n'est
    necessaire : NumPy stocke deja (N, C, H, W) apres le transpose(0, 3, 1, 2)
    fait dans les scripts.
    """
    d = np.load(chemin)
    cles = list(d.keys())
    convs = (_trier_cles(cles, "conv_", "_kernels"), _trier_cles(cles, "conv_", "_bias"))
    denses = (_trier_cles(cles, "dense_", "_W"), _trier_cles(cles, "dense_", "_b"))

    if len(convs[0]) != 2 or len(denses[0]) != 2:
        raise ValueError(f"fichier .npz inattendu : {cles}")

    with torch.no_grad():
        for module, k_kernels, k_bias in zip((modele.conv1, modele.conv2),
                                             convs[0], convs[1]):
            module.weight.copy_(torch.from_numpy(d[k_kernels].astype(np.float32)))
            module.bias.copy_(torch.from_numpy(d[k_bias].astype(np.float32)).flatten())
        for module, k_W, k_b in zip((modele.fc1, modele.fc2), denses[0], denses[1]):
            module.weight.copy_(torch.from_numpy(d[k_W].astype(np.float32)))
            module.bias.copy_(torch.from_numpy(d[k_b].astype(np.float32)).flatten())
    return modele


def sauver_npz(modele, chemin):
    """Sauvegarde au format .npz de la version NumPy (interchangeable).

    On reprend les indices de couche de l'architecture NumPy : conv 0 et 3,
    dense 7 et 9. Le biais est remis en forme (C, 1) comme dans la version NumPy.
    """
    d = {}
    with torch.no_grad():
        for i, module in ((0, modele.conv1), (3, modele.conv2)):
            d[f"conv_{i}_kernels"] = module.weight.detach().cpu().numpy()
            d[f"conv_{i}_bias"] = module.bias.detach().cpu().numpy().reshape(-1, 1)
        for i, module in ((7, modele.fc1), (9, modele.fc2)):
            d[f"dense_{i}_W"] = module.weight.detach().cpu().numpy()
            d[f"dense_{i}_b"] = module.bias.detach().cpu().numpy().reshape(-1, 1)
    np.savez(chemin, **d)
    return chemin


def nb_parametres(modele):
    return sum(p.numel() for p in modele.parameters())
