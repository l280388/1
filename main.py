#!/usr/bin/env python3
"""
Geometry Mass Calculator — Interfaccia a linea di comando
==========================================================
Consente di costruire sezioni trasversali complesse aggiungendo o
sottraendo forme geometriche di base e di calcolarne le proprietà:
  • Area
  • Momenti statici (Sx, Sy)
  • Baricentro (xc, yc)
  • Momenti d'inerzia rispetto all'origine (Ix, Iy, Ixy)
  • Momenti d'inerzia baricentrici (Ixc, Iyc, Ixyc)
  • Raggi di girazione (ix, iy)

Avvio: python main.py
"""

import os
import sys

from geometry_calculator import (
    CrossSection,
    Rectangle,
    Circle,
    HollowCircle,
    Triangle,
    Semicircle,
    Polygon,
)
from image_analyzer import ContourAnalyzer, ClaudeVisionAnalyzer, visualize_section


# ─── Helper I/O ──────────────────────────────────────────────────────────────

def clr():
    """Pulisce lo schermo (cross-platform)."""
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    print("╔══════════════════════════════════════════════════════╗")
    print("║       GEOMETRY MASS CALCULATOR  v1.0                ║")
    print("║  Area · Momenti Statici · Momenti d'Inerzia          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def ask_float(prompt: str, positive: bool = False) -> float:
    """Chiede un numero reale all'utente con validazione."""
    while True:
        raw = input(prompt).strip().replace(",", ".")
        try:
            v = float(raw)
            if positive and v <= 0:
                print("  ⚠  Il valore deve essere positivo.")
                continue
            return v
        except ValueError:
            print("  ⚠  Inserire un numero valido.")


def ask_int(prompt: str, lo: int = 0, hi: int = 9999) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
            print(f"  ⚠  Inserire un valore tra {lo} e {hi}.")
        except ValueError:
            print("  ⚠  Inserire un numero intero.")


def ask_name(default: str) -> str:
    raw = input(f"  Nome forma [{default}]: ").strip()
    return raw if raw else default


def ask_operation() -> int:
    """Chiede se aggiungere (+1) o sottrarre (−1) la forma."""
    print("  Operazione:")
    print("    1. Aggiungere (+)  — materiale pieno")
    print("    2. Sottrarre  (−)  — foro / cavità")
    return 1 if ask_int("  Scelta [1/2]: ", 1, 2) == 1 else -1


def pause():
    input("\n  Premi INVIO per continuare...")


# ─── Creazione delle forme ────────────────────────────────────────────────────

def create_rectangle() -> Rectangle:
    print("\n  ── RETTANGOLO ─────────────────────────────────────")
    print("  Il punto di riferimento è l'angolo inferiore sinistro.")
    name = ask_name("Rettangolo")
    b = ask_float("  Base  b : ", positive=True)
    h = ask_float("  Altezza h : ", positive=True)
    x0 = ask_float("  Angolo inf. sin. X (x0) [0]: ") if input("  Posizione non nulla? [s/N] ").lower() == "s" else 0.0
    y0 = ask_float("  Angolo inf. sin. Y (y0) [0]: ") if input("  Posizione non nulla? [s/N] ").lower() == "s" else 0.0
    return Rectangle(b, h, x0, y0, name=name)


def create_circle() -> Circle:
    print("\n  ── CERCHIO ────────────────────────────────────────")
    name = ask_name("Cerchio")
    r = ask_float("  Raggio r : ", positive=True)
    xc = ask_float("  Centro X (xc) [0]: ") if input("  Centro non nell'origine? [s/N] ").lower() == "s" else 0.0
    yc = ask_float("  Centro Y (yc) [0]: ") if input("  (yc) non nulla? [s/N] ").lower() == "s" else 0.0
    return Circle(r, xc, yc, name=name)


def create_hollow_circle() -> HollowCircle:
    print("\n  ── ANELLO CIRCOLARE ───────────────────────────────")
    name = ask_name("Anello")
    R = ask_float("  Raggio esterno R : ", positive=True)
    r = ask_float("  Raggio interno r : ", positive=True)
    xc = ask_float("  Centro X (xc) [0]: ") if input("  Centro non nell'origine? [s/N] ").lower() == "s" else 0.0
    yc = ask_float("  Centro Y (yc) [0]: ") if input("  (yc) non nulla? [s/N] ").lower() == "s" else 0.0
    return HollowCircle(R, r, xc, yc, name=name)


def create_triangle() -> Triangle:
    print("\n  ── TRIANGOLO RETTANGOLO ───────────────────────────")
    print("  Il vertice retto è in (x0, y0).")
    print("  La base si estende verso +x, l'altezza verso +y.")
    name = ask_name("Triangolo")
    b = ask_float("  Base  b : ", positive=True)
    h = ask_float("  Altezza h : ", positive=True)
    x0 = ask_float("  Vertice retto X (x0) [0]: ") if input("  Posizione non nulla? [s/N] ").lower() == "s" else 0.0
    y0 = ask_float("  Vertice retto Y (y0) [0]: ") if input("  Posizione non nulla? [s/N] ").lower() == "s" else 0.0
    return Triangle(b, h, x0, y0, name=name)


def create_semicircle() -> Semicircle:
    print("\n  ── SEMICERCHIO ────────────────────────────────────")
    print("  Il lato piatto è rivolto verso il basso.")
    print("  Il centro del diametro piatto è in (x0, y0).")
    name = ask_name("Semicerchio")
    r = ask_float("  Raggio r : ", positive=True)
    x0 = ask_float("  Centro diametro X (x0) [0]: ") if input("  Posizione non nulla? [s/N] ").lower() == "s" else 0.0
    y0 = ask_float("  Centro diametro Y (y0) [0]: ") if input("  Posizione non nulla? [s/N] ").lower() == "s" else 0.0
    return Semicircle(r, x0, y0, name=name)


def create_polygon() -> Polygon:
    print("\n  ── POLIGONO GENERICO ──────────────────────────────")
    print("  Inserire i vertici in ordine ANTIORARIO (CCW).")
    name = ask_name("Poligono")
    n = ask_int("  Numero di vertici (≥ 3): ", lo=3)
    vertices = []
    for i in range(n):
        print(f"  Vertice {i + 1}:")
        x = ask_float(f"    X{i + 1} : ")
        y = ask_float(f"    Y{i + 1} : ")
        vertices.append((x, y))
    return Polygon(vertices, name=name)


SHAPE_CREATORS = {
    "1": ("Rettangolo", create_rectangle),
    "2": ("Cerchio", create_circle),
    "3": ("Anello circolare (cavo)", create_hollow_circle),
    "4": ("Triangolo rettangolo", create_triangle),
    "5": ("Semicerchio", create_semicircle),
    "6": ("Poligono generico", create_polygon),
}


def menu_add_shape(section: CrossSection) -> None:
    print("\n  Scegli il tipo di forma:")
    for key, (label, _) in SHAPE_CREATORS.items():
        print(f"    {key}. {label}")
    print("    0. Annulla")
    choice = input("  Scelta: ").strip()
    if choice == "0" or choice not in SHAPE_CREATORS:
        return
    _, creator = SHAPE_CREATORS[choice]
    try:
        shape = creator()
        sign = ask_operation()
        shape.sign = sign
        section.shapes.append(shape)
        op = "aggiunto" if sign == 1 else "sottratto"
        print(f"\n  ✓ '{shape.name}' {op} alla sezione.")
    except ValueError as e:
        print(f"\n  ✗ Errore: {e}")
    pause()


# ─── Menu principale ──────────────────────────────────────────────────────────

def menu_list(section: CrossSection) -> None:
    print()
    if not section.shapes:
        print("  La sezione è vuota.")
    else:
        print(f'  Sezione: "{section.name}"  ({len(section.shapes)} forma/e)\n')
        print(f"  {'#':>3}  {'Op':^4}  {'Tipo':<22}  {'Nome':<18}  {'Area':>14}  {'Baricentro (xc, yc)'}")
        print("  " + "─" * 80)
        for i, s in enumerate(section.shapes):
            op = " + " if s.sign == 1 else " − "
            xc, yc = s.centroid()
            print(
                f"  {i:3d}  {op:^4}  {s.__class__.__name__:<22}  {s.name:<18}  "
                f"{s.area():>14.4f}  ({xc:.4f}, {yc:.4f})"
            )
    pause()


def menu_calculate(section: CrossSection) -> None:
    print()
    try:
        props = section.calculate()
        print(props.summary(f'SEZIONE: "{section.name}"'))
    except ValueError as e:
        print(f"  ✗ {e}")
    pause()


def menu_remove(section: CrossSection) -> None:
    if not section.shapes:
        print("  La sezione è vuota.")
        pause()
        return
    print()
    for i, s in enumerate(section.shapes):
        op = "+" if s.sign == 1 else "−"
        print(f"  {i:3d}.  [{op}]  {s.name}  ({s.__class__.__name__})")
    idx = ask_int("  Indice da rimuovere (−1 per annullare): ", lo=-1, hi=len(section.shapes) - 1)
    if idx == -1:
        return
    removed = section.shapes[idx].name
    section.remove(idx)
    print(f"  ✓ '{removed}' rimosso.")
    pause()


def menu_rename(section: CrossSection) -> None:
    raw = input(f'  Nuovo nome per la sezione [{section.name}]: ').strip()
    if raw:
        section.name = raw
        print(f"  ✓ Sezione rinominata in \"{section.name}\".")
    pause()


def menu_save(section: CrossSection) -> None:
    fname = input("  Nome file (es. sezione.json): ").strip()
    if not fname:
        print("  ✗ Nome file non valido.")
        pause()
        return
    if not fname.endswith(".json"):
        fname += ".json"
    try:
        section.save(fname)
        print(f"  ✓ Sezione salvata in '{fname}'.")
    except OSError as e:
        print(f"  ✗ Errore: {e}")
    pause()


def menu_load() -> CrossSection | None:
    fname = input("  Nome file da caricare: ").strip()
    try:
        sec = CrossSection.load(fname)
        print(f"  ✓ Sezione '{sec.name}' caricata ({len(sec.shapes)} forme).")
        pause()
        return sec
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"  ✗ Errore: {e}")
        pause()
        return None


def menu_examples() -> CrossSection | None:
    print("\n  Esempi predefiniti:")
    print("    1. Sezione a T (flangia + anima)")
    print("    2. Sezione IPE semplificata")
    print("    3. Sezione rettangolare cava")
    print("    4. Sezione circolare cava (tubo)")
    print("    5. Sezione L (angolare)")
    print("    0. Torna al menu principale")
    choice = input("  Scelta: ").strip()
    sec = None
    if choice == "1":
        sec = _example_T()
    elif choice == "2":
        sec = _example_IPE()
    elif choice == "3":
        sec = _example_hollow_rect()
    elif choice == "4":
        sec = _example_tube()
    elif choice == "5":
        sec = _example_L()
    if sec:
        print()
        props = sec.calculate()
        print(props.summary(f'ESEMPIO: "{sec.name}"'))
        print()
        keep = input("  Vuoi lavorare su questa sezione? [s/N] ").lower()
        if keep == "s":
            return sec
    pause()
    return None


# ─── Esempi predefiniti ───────────────────────────────────────────────────────

def _example_T() -> CrossSection:
    """Sezione a T: flangia 200×20 + anima 20×180."""
    sec = CrossSection("Sezione a T")
    sec.add(Rectangle(200, 20, 0, 180, name="Flangia"))   # flangia in cima
    sec.add(Rectangle(20, 180, 90, 0, name="Anima"))      # anima al centro
    return sec


def _example_IPE() -> CrossSection:
    """Sezione IPE semplificata: 2 flange + anima."""
    bf, tf = 100.0, 8.5    # flangia: larghezza, spessore
    hw, tw = 183.0, 5.7    # anima: altezza, spessore
    h_tot = hw + 2 * tf
    sec = CrossSection("IPE 200 semplificata")
    sec.add(Rectangle(bf, tf, -(bf / 2), 0, name="Flangia inf."))
    sec.add(Rectangle(tw, hw, -(tw / 2), tf, name="Anima"))
    sec.add(Rectangle(bf, tf, -(bf / 2), tf + hw, name="Flangia sup."))
    return sec


def _example_hollow_rect() -> CrossSection:
    """Sezione rettangolare cava: 200×300 con foro 160×260."""
    sec = CrossSection("Rettangolare cava")
    sec.add(Rectangle(200, 300, 0, 0, name="Esterno"))
    sec.subtract(Rectangle(160, 260, 20, 20, name="Foro interno"))
    return sec


def _example_tube() -> CrossSection:
    """Tubo circolare: D=100, d=80."""
    sec = CrossSection("Tubo circolare")
    sec.add(HollowCircle(50, 40, 0, 0, name="Tubo D=100 d=80"))
    return sec


def _example_L() -> CrossSection:
    """Angolare L 100×100×10."""
    sec = CrossSection("Angolare L 100×100×10")
    sec.add(Rectangle(100, 10, 0, 0, name="Ala orizzontale"))
    sec.add(Rectangle(10, 90, 0, 10, name="Ala verticale"))
    return sec


# ─── Menu: importa da immagine ───────────────────────────────────────────────

def menu_import_image() -> CrossSection | None:
    print("\n  ── IMPORTA DA IMMAGINE ────────────────────────────")
    print("  Metodi disponibili:")
    print("    1. Analisi contorno (OpenCV)")
    print("       Estrae il profilo da silhouette / disegni B&N.")
    print("       Non richiede connessione internet.")
    print()
    print("    2. Analisi AI (Claude Vision)")
    print("       Interpreta disegni tecnici quotati, schizzi, foto.")
    print("       Richiede ANTHROPIC_API_KEY.")
    print()
    print("    0. Annulla")

    choice = input("  Metodo: ").strip()
    if choice == "0":
        return None

    img_path = input("  Percorso immagine: ").strip().strip('"').strip("'")
    if not img_path:
        print("  ✗ Percorso non valido.")
        pause()
        return None

    # ── Metodo 1: contorno OpenCV ──────────────────────────────────────
    if choice == "1":
        print("\n  Opzioni avanzate (premi INVIO per i valori predefiniti):")
        scale_s = input("  Scala (mm/pixel) [1.0]: ").strip().replace(",", ".")
        scale = float(scale_s) if scale_s else 1.0

        invert_s = input("  Forma chiara su sfondo scuro? [s/N]: ").strip().lower()
        invert = invert_s == "s"

        thresh_s = input("  Soglia binarizzazione 0-255 [127]: ").strip()
        threshold = int(thresh_s) if thresh_s.isdigit() else 127

        preview_s = input("  Mostra anteprima contorni? [S/n]: ").strip().lower()
        do_preview = preview_s != "n"

        try:
            analyzer = ContourAnalyzer()
            if do_preview:
                print("  Apertura anteprima...")
                analyzer.preview(img_path, scale=scale, invert=invert,
                                 threshold=threshold)
            sec = analyzer.analyze(img_path, scale=scale, invert=invert,
                                   threshold=threshold)
            print(f"\n  ✓ Estratte {len(sec.shapes)} forma/e.")
            props = sec.calculate()
            print(props.summary(f'SEZIONE DA IMMAGINE: "{sec.name}"'))
            keep = input("\n  Usare questa sezione? [S/n]: ").strip().lower()
            if keep != "n":
                pause()
                return sec
        except Exception as e:
            print(f"\n  ✗ Errore: {e}")

    # ── Metodo 2: Claude Vision API ────────────────────────────────────
    elif choice == "2":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            api_key = input("  ANTHROPIC_API_KEY: ").strip()

        try:
            analyzer = ClaudeVisionAnalyzer(api_key=api_key)
            sec = analyzer.analyze(img_path)
            print(f"\n  ✓ Sezione '{sec.name}' ricostruita da Claude.")
            props = sec.calculate()
            print(props.summary(f'SEZIONE DA AI: "{sec.name}"'))
            keep = input("\n  Usare questa sezione? [S/n]: ").strip().lower()
            if keep != "n":
                pause()
                return sec
        except Exception as e:
            print(f"\n  ✗ Errore: {e}")

    else:
        print("  ⚠  Scelta non valida.")

    pause()
    return None


def menu_visualize(section: CrossSection) -> None:
    if not section.shapes:
        print("  Sezione vuota: aggiungere prima delle forme.")
        pause()
        return
    print("  Apertura visualizzazione grafica...")
    try:
        visualize_section(section)
    except Exception as e:
        print(f"  ✗ Errore visualizzazione: {e}")
        pause()


# ─── Loop principale ──────────────────────────────────────────────────────────

def main():
    section = CrossSection("Nuova sezione")

    while True:
        clr()
        banner()
        print(f'  Sezione corrente: "{section.name}"  '
              f'({len(section.shapes)} forma/e)')
        print()
        print("  ┌─ FORME ──────────────────────────────────────────┐")
        print("  │  1. Aggiungi / sottrai forma                     │")
        print("  │  2. Elenca forme                                 │")
        print("  │  3. Rimuovi una forma                            │")
        print("  │  4. Svuota la sezione                            │")
        print("  ├─ CALCOLO ────────────────────────────────────────┤")
        print("  │  5. Calcola proprietà della sezione              │")
        print("  ├─ FILE ───────────────────────────────────────────┤")
        print("  │  6. Salva sezione (JSON)                         │")
        print("  │  7. Carica sezione (JSON)                        │")
        print("  │  8. Rinomina sezione                             │")
        print("  ├─ IMMAGINI ───────────────────────────────────────┤")
        print("  │  i. Importa da immagine (contorno / AI)          │")
        print("  │  v. Visualizza sezione corrente                  │")
        print("  ├─ EXTRA ──────────────────────────────────────────┤")
        print("  │  9. Esempi predefiniti                           │")
        print("  │  0. Esci                                         │")
        print("  └──────────────────────────────────────────────────┘")
        print()

        choice = input("  Scelta: ").strip()

        if choice == "1":
            menu_add_shape(section)
        elif choice == "2":
            menu_list(section)
        elif choice == "3":
            menu_remove(section)
        elif choice == "4":
            section.clear()
            print("  Sezione svuotata.")
            pause()
        elif choice == "5":
            menu_calculate(section)
        elif choice == "6":
            menu_save(section)
        elif choice == "7":
            loaded = menu_load()
            if loaded:
                section = loaded
        elif choice == "8":
            menu_rename(section)
        elif choice == "9":
            result = menu_examples()
            if result:
                section = result
        elif choice in ("i", "I"):
            result = menu_import_image()
            if result:
                section = result
        elif choice in ("v", "V"):
            menu_visualize(section)
        elif choice == "0":
            print("\n  Arrivederci!\n")
            sys.exit(0)
        else:
            print("  ⚠  Scelta non valida.")
            pause()


if __name__ == "__main__":
    main()
