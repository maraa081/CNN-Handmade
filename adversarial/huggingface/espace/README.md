---
title: Robustesse adversariale - attaque PGD sur MNIST
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Demo : un modele robuste face a PGD

Espace de demonstration du depot
[CNN-Handmade](https://github.com/maraa081/CNN-Handmade).

Choisis un chiffre, un budget d'attaque `eps` et un modele : le Space predit
l'image d'origine, lance une attaque PGD, puis predit l'image perturbee. Le
temoin non defendu tombe immediatement ; les modeles entraines avec un plan de
budget croissant tiennent nettement mieux.

Les chiffres d'annonce (AutoAttack, 10 000 images, eps=0.30) sont dans la carte
du modele : ici on ne mesure rien sur une image, on montre.
