"""Espace de demonstration : le modele robuste face a une attaque PGD.

Ce que montre l'espace, en une interaction : on choisit une image de chiffre, on
choisit un budget d'attaque `eps`, et on regarde DEUX modeles predire la meme
image perturbee -- un modele entraine proprement (il tombe tout de suite) et un
modele entraine contre PGD avec un plan de budget croissant (il tient).

But pedagogique, pas benchmark : les chiffres d'annonce (AutoAttack, 10 000
images, eps=0.30) sont dans la carte du modele, pas ici. Un Space qui montrerait
une precision calculee sur une image ne prouverait rien.

Reglages a confirmer avant publication : `MODELE_HF` (depot de poids) et
`FICHIERS` (noms des fichiers pousses). Voir `../REGLAGES.md`.
"""

import os

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

# --------------------------------------------------------------------------
#  Configuration
# --------------------------------------------------------------------------

MODELE_HF = os.environ.get("MODELE_HF", "maraa081/mnist-cnn-robuste")

# Libelle affiche -> fichier de poids dans le depot Hugging Face.
FICHIERS = {
    "Robuste : budget adaptatif par batch (91.25% officiel)": "bande_cible50.pt",
    "Robuste : plan 0.2 -> 2 eps (82.40% officiel)": "a6_plan_budget.pt",
    "Robuste : plan 0.2 -> 1 eps (78.25% officiel)": "a8_plan_0p2_1.pt",
    "Reference : constante 2 eps, depart dur (50.71% officiel)": "abl_a_eps10.pt",
    "Non defendu (temoin)": "standard.pt",
}

# Normalisation utilisee a l'entrainement (MNIST : moyenne et ecart-type du jeu).
MOYENNE, ECART = 0.1307, 0.3081

# Le budget maximal affiche. Les modeles sont entraines a eps=0.30 en L-infini :
# au-dela, on sort du regime qu'ils connaissent (et le Space n'aurait plus de
# sens : n'importe quel modele casse a eps=0.5).
EPS_MAX = 0.30


# --------------------------------------------------------------------------
#  Le modele (meme architecture que src/model.py du depot)
# --------------------------------------------------------------------------

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


_CACHE = {}


def charger(libelle):
    """Charge (et memorise) un modele depuis le depot de poids."""
    if libelle in _CACHE:
        return _CACHE[libelle]
    fichier = FICHIERS[libelle]
    chemin = hf_hub_download(repo_id=MODELE_HF, filename=fichier)
    ck = torch.load(chemin, map_location="cpu", weights_only=False)
    etat = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    modele = CNN()
    modele.load_state_dict(etat)
    modele.eval()
    _CACHE[libelle] = modele
    return modele


# --------------------------------------------------------------------------
#  Image -> tenseur
# --------------------------------------------------------------------------

def preparer(image):
    """Image (numpy, dessinee ou chargee) -> tenseur (1, 1, 28, 28) normalise.

    Trois pieges classiques dans ce genre de demo :

    1. le sens des couleurs. Sur un pad de dessin, le trait est CLAIR sur fond
       SOMBRE ; sur une photo de chiffre, c'est l'inverse. On regarde la
       mediane : si elle est claire, l'image est inversee (fond blanc), et on
       la remet a l'endroit attendu par le modele (chiffre clair, fond sombre).
    2. la taille. `gr.Image` peut rendre du 200x200 : on redimensionne en
       28x28 avec PIL (et pas avec un slicing grossier).
    3. la normalisation. Meme moyenne et meme ecart-type qu'a l'entrainement --
       sinon un modele parfait predit n'importe quoi.
    """
    from PIL import Image

    if image is None:
        raise gr.Error("Charge ou dessine d'abord un chiffre.")

    tab = np.asarray(image)
    if tab.ndim == 3:
        tab = tab[..., :3].mean(axis=2)          # RVB -> niveaux de gris
    tab = tab.astype(np.float32)

    if tab.max() <= 1.0:                          # image deja dans [0, 1]
        tab *= 255.0
    if np.median(tab) > 127:                      # fond clair -> on inverse
        tab = 255.0 - tab

    petit = np.asarray(Image.fromarray(tab.astype(np.uint8)).resize((28, 28),
                                                                    Image.BILINEAR))
    x = torch.tensor(petit, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    x = (x - MOYENNE) / ECART
    return x


# --------------------------------------------------------------------------
#  L'attaque
# --------------------------------------------------------------------------

def pgd_ce(modele, x, y, eps, pas):
    """PGD sur l'entropie croisee, pas `eps/4` (la reference du depot).

    `y` est la classe predite par le modele PROPRE sur l'image non perturbee :
    c'est l'etiquette qu'on utilise, exactement comme dans un banc d'essai ou
    l'on ne trichait pas avec la vraie etiquette du jeu de donnees.
    """
    alpha = eps / 4.0
    x_adv = x.clone().detach()
    for _ in range(pas):
        x_adv.requires_grad_(True)
        perte = nn.functional.cross_entropy(modele(x_adv), y)
        grad = torch.autograd.grad(perte, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
    return x_adv.detach()


def vers_image(tenseur):
    """Tenseur normalise -> image [0, 1] affichable."""
    tab = tenseur.detach().squeeze().numpy() * ECART + MOYENNE
    return np.clip(tab, 0, 1)


# --------------------------------------------------------------------------
#  Le callback
# --------------------------------------------------------------------------

def analyser(image, libelle, eps, pas):
    if image is None:
        raise gr.Error("Charge ou dessine d'abord un chiffre.")
    if eps == 0:
        pas = 0

    modele = charger(libelle)
    x = preparer(image)

    with torch.no_grad():
        logits = modele(x)
        probs = torch.softmax(logits, dim=1)[0]
        classe = int(probs.argmax())
    etiquette = torch.tensor([classe])

    if pas == 0:
        x_adv = x
    else:
        x_adv = pgd_ce(modele, x, etiquette, eps, int(pas))

    with torch.no_grad():
        probs_adv = torch.softmax(modele(x_adv), dim=1)[0]

    classes = [str(i) for i in range(10)]
    propre = {c: float(p) for c, p in zip(classes, probs)}
    adverse = {c: float(p) for c, p in zip(classes, probs_adv)}

    classe_adv = int(probs_adv.argmax())
    if pas == 0:
        verdict = "Aucune attaque (budget nul) : les deux predictions sont identiques."
    elif classe_adv != classe:
        verdict = (f"**Le modele est trompe** : `{classe}` devient `{classe_adv}` "
                   f"a eps={eps:.2f} en {int(pas)} pas de PGD.")
    else:
        verdict = (f"**Le modele tient** : il predit encore `{classe}` "
                   f"(confiance {float(probs_adv[classe]):.1%}) a eps={eps:.2f}.")

    # Perturbation : amplifiee x10 et ramenee dans [0, 1] pour etre visible.
    delta = (x_adv - x).squeeze().numpy()
    perturbation = np.clip(0.5 + delta * 10.0 / (2 * EPS_MAX), 0, 1)

    return propre, adverse, vers_image(x_adv), perturbation, verdict


# --------------------------------------------------------------------------
#  Interface
# --------------------------------------------------------------------------

EXEMPLES = []
dossier_exemples = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exemples")
if os.path.isdir(dossier_exemples):
    for i in range(10):
        chemin = os.path.join(dossier_exemples, f"chiffre_{i}.png")
        if os.path.exists(chemin):
            EXEMPLES.append([chemin])

with gr.Blocks(title="Robustesse adversariale : PGD sur MNIST") as demo:
    gr.Markdown(
        "# Un modele robuste, en direct\n"
        "Choisis un chiffre, choisis un budget d'attaque `eps` (L-infini, 0 a 0.30), "
        "et regarde le meme modele predire l'image avant et apres une attaque PGD.\n\n"
        "Les chiffres d'annonce viennent d'AutoAttack sur 10 000 images (voir la "
        "carte du modele). Ici on ne mesure rien : on montre."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Chiffre (dessine en clair sur fond sombre)",
                             image_mode="L", height=280)
            modele_nom = gr.Dropdown(list(FICHIERS), value=list(FICHIERS)[0],
                                     label="Modele")
            eps = gr.Slider(0.0, EPS_MAX, value=0.15, step=0.01,
                            label="Budget d'attaque eps (L-infini)")
            pas = gr.Slider(0, 40, value=20, step=1,
                            label="Pas de PGD (0 = aucune attaque)")
            bouton = gr.Button("Attaquer", variant="primary")
            if EXEMPLES:
                gr.Examples(examples=EXEMPLES, inputs=[image],
                            label="Exemples (jeu de test MNIST)")
        with gr.Column():
            verdict = gr.Markdown()
            avec_attaque = gr.Label(label="Apres attaque (image perturbee)",
                                    num_top_classes=3)
            propre = gr.Label(label="Sur l'image d'origine", num_top_classes=3)
            image_adv = gr.Image(label="Image perturbee", height=200)
            perturbation = gr.Image(
                label="Perturbation (amplifiee x10 : le bruit invisible)",
                height=200)

    bouton.click(analyser, [image, modele_nom, eps, pas],
                 [propre, avec_attaque, image_adv, perturbation, verdict])

if __name__ == "__main__":
    demo.launch()
