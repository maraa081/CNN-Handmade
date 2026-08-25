#!/usr/bin/env python3
"""Nettoyage emojis + unicode décoratif dans le repo CNN-Handmade.

Règles :
- Emojis (pictogrammes, dingbats, symboles divers) -> mot ASCII ou suppression
- Traits de boîte (═ ─ │ ├ ...) -> ASCII (| - + = `)
- Flèches (→ ← ↔) -> -> <- <->
- Sélecteurs de variation + keycaps -> supprimés
- GARDE : accents français (é è à ç ...), lettres grecques (θ ε α β λ ∇),
  ponctuation française (— … « »), symboles math (² ₁ ₂ ± ≈ ≤ ° √ ∈ ∞)

Usage : python3 scripts_cleanup_emojis.py
"""
import subprocess
import unicodedata

EMOJI_WORDS = {
    "\u2705": "OK",        # ✅
    "\u274c": "FAIL",      # ❌
    "\u2714": "OK",        # ✔
    "\u26a0": "[warn]",    # ⚠
    "\u23f1": "[time]",    # ⏱
    "\u23f3": "[wait]",    # ⏳
    "\u23ed": ">>",        # ⏭
    "\u25b6": ">",         # ▶
}

BOX = {
    "\u2500": "-", "\u2501": "-",           # ─ ━
    "\u2502": "|", "\u2503": "|",           # │ ┃
    "\u2550": "=", "\u2551": "|",           # ═ ║
    "\u251c": "|",                          # ├
    "\u2514": "`",                          # └
    "\u250c": "+", "\u2510": "+",           # ┌ ┐
    "\u2518": "+", "\u2524": "+",           # ┘ ┤
    "\u252c": "+", "\u2534": "+",           # ┬ ┴
    "\u253c": "+",                          # ┼
    "\u2554": "+", "\u2557": "+",           # ╔ ╗
    "\u255a": "+", "\u255d": "+",           # ╚ ╝
    "\u2560": "+", "\u2563": "+",           # ╠ ╣
    "\u2566": "+", "\u2569": "+",           # ╦ ╩
    "\u256c": "+",                          # ╬
}

ARROWS = {
    "\u2190": "<-", "\u2192": "->", "\u2194": "<->",
    "\u2191": "^", "\u2193": "v",
    "\u21d0": "<=", "\u21d2": "=>",
}

# Espaces / artefacts
MISC = {
    "\u202f": " ",       # espace fine insécable
    "\u0302": "",        # accent circonflexe combinant (artefact)
}

def is_emoji(c):
    o = ord(c)
    return (0x1F000 <= o <= 0x1FAFF or   # pictogrammes
            0x2600 <= o <= 0x27BF or      # symboles divers + dingbats
            0xFE00 <= o <= 0xFE0F or      # sélecteurs de variation
            o == 0x20E3)                  # keycap combinant

def clean(text, keep_fr=True):
    out = []
    for c in text:
        if c in EMOJI_WORDS:
            out.append(EMOJI_WORDS[c])
        elif c in BOX:
            out.append(BOX[c])
        elif c in ARROWS:
            out.append(ARROWS[c])
        elif c in MISC:
            out.append(MISC[c])
        elif is_emoji(c):
            continue  # suppression
        else:
            out.append(c)
    return "".join(out)

def main():
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split()
    changed = 0
    for f in files:
        try:
            txt = open(f, encoding="utf-8").read()
        except Exception:
            continue
        new = clean(txt)
        if new != txt:
            open(f, "w", encoding="utf-8").write(new)
            print(f"  nettoyé: {f}")
            changed += 1
    print(f"\n{changed} fichiers modifiés")

if __name__ == "__main__":
    main()
