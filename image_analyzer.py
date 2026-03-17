"""
Image Analyzer per Geometry Mass Calculator
============================================
Estrae le proprietà geometriche di sezioni trasversali a partire da immagini,
attraverso due metodi complementari:

  1. ContourAnalyzer  — computer vision (OpenCV)
     Rileva il profilo della sezione da silhouette / disegni in bianco e nero.
     Restituisce un Polygon (o più poligoni per fori interni) pronto per
     l'integrazione nella CrossSection.

  2. ClaudeVisionAnalyzer  — intelligenza artificiale (Claude Vision API)
     Interpreta disegni tecnici quotati, schizzi, foto di sezioni.
     Claude riconosce forme, dimensioni e posizioni, restituisce un JSON
     strutturato che viene trasformato automaticamente in CrossSection.

Dipendenze:
  pip install opencv-python numpy anthropic matplotlib
"""

import base64
import json
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from geometry_calculator import (
    CrossSection, Polygon,
    Rectangle, Circle, HollowCircle, Triangle, Semicircle,
    _shape_from_dict,
)


# ─── 1. Analisi contorno (OpenCV) ────────────────────────────────────────────

class ContourAnalyzer:
    """
    Estrae il profilo di una sezione da un'immagine tramite rilevamento
    del contorno con OpenCV.

    Funziona meglio con:
      • disegni tecnici in bianco e nero
      • silhouette con buon contrasto
      • foto di sezioni con sfondo uniforme

    Il sistema di coordinate viene ribaltato verticalmente (OpenCV mette y=0
    in alto; la geometria strutturale lo vuole in basso).
    """

    def analyze(
        self,
        image_path: str,
        scale: float = 1.0,
        epsilon_factor: float = 0.003,
        invert: bool = False,
        threshold: int = 127,
        min_area_px: int = 200,
    ) -> CrossSection:
        """
        Rileva tutti i contorni significativi e costruisce la CrossSection.

        Args:
            image_path:     percorso all'immagine
            scale:          fattore di scala pixel→unità reali (es. mm/pixel)
            epsilon_factor: tolleranza approssimazione poligono (0.001–0.01)
            invert:         True se la forma è chiara su sfondo scuro
            threshold:      soglia per la binarizzazione (0–255)
            min_area_px:    area minima in pixel per includere un contorno

        Returns:
            CrossSection con profilo esterno + eventuali fori interni
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Impossibile aprire l'immagine: {image_path}")

        h_img = img.shape[0]

        # ── Pre-processing ────────────────────────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh_type = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
        _, binary = cv2.threshold(gray, threshold, 255, thresh_type)

        # Morfologia: chiude piccoli buchi, rimuove rumore
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

        # ── Rilevamento contorni con gerarchia ───────────────────────────
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError("Nessun contorno trovato nell'immagine.")

        hierarchy = hierarchy[0]   # shape: (N, 4)  [next, prev, child, parent]
        section = CrossSection(Path(image_path).stem)
        shape_count = 0

        for idx, (cnt, hier) in enumerate(zip(contours, hierarchy)):
            area_px = cv2.contourArea(cnt)
            if area_px < min_area_px:
                continue

            parent_idx = hier[3]   # -1 → contorno radice (esterno)

            # Approssima il contorno come poligono
            peri = cv2.arcLength(cnt, True)
            epsilon = epsilon_factor * peri
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Converti pixel → unità reali; ribalta Y
            pts = [
                (float(p[0][0]) * scale, float(h_img - p[0][1]) * scale)
                for p in approx
            ]
            if len(pts) < 3:
                continue

            try:
                poly = Polygon(pts, name=f"Profilo-{shape_count}")
                shape_count += 1
                if parent_idx >= 0:
                    # Contorno annidato → foro/cavità
                    section.subtract(poly)
                else:
                    section.add(poly)
            except ValueError:
                continue

        if not section.shapes:
            raise ValueError(
                "Nessuna forma valida estratta. "
                "Prova a modificare threshold o invert."
            )
        return section

    def preview(self, image_path: str, **kwargs) -> None:
        """
        Mostra l'immagine con i contorni rilevati sovrapposti
        (utile per verificare l'estrazione prima del calcolo).
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(image_path)

        # Ottieni la sezione per estrarre i poligoni
        sec = self.analyze(image_path, **kwargs)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h = img_bgr.shape[0]
        scale = kwargs.get("scale", 1.0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img_rgb)
        axes[0].set_title("Immagine originale")
        axes[0].axis("off")

        axes[1].set_title("Contorni rilevati")
        axes[1].set_aspect("equal")
        axes[1].invert_yaxis()

        colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
        for i, shape in enumerate(sec.shapes):
            if isinstance(shape, Polygon):
                xs = [v[0] for v in shape.vertices] + [shape.vertices[0][0]]
                ys = [v[1] for v in shape.vertices] + [shape.vertices[0][1]]
                color = colors[i % len(colors)]
                lbl = f"{'+ ' if shape.sign==1 else '− '}{shape.name}"
                axes[1].plot(xs, ys, color=color, linewidth=2, label=lbl)
                axes[1].fill(
                    [v[0] for v in shape.vertices],
                    [v[1] for v in shape.vertices],
                    alpha=0.15, color=color
                )

        axes[1].legend(fontsize=8)
        axes[1].set_xlabel(f"x  (scala 1 px = {scale} unità)")
        axes[1].set_ylabel("y")
        plt.tight_layout()
        plt.show()


# ─── 2. Analisi AI con Claude Vision ────────────────────────────────────────

_CLAUDE_SYSTEM = """Sei un esperto di geometria strutturale e disegno tecnico.

Analizza l'immagine fornita (disegno tecnico, schizzo, foto di sezione trasversale)
e identifica tutte le forme geometriche che la compongono.

Rispondi ESCLUSIVAMENTE con un oggetto JSON valido (senza blocchi markdown, senza testo aggiuntivo):

{
  "section_name": "nome descrittivo della sezione",
  "unit": "mm",
  "shapes": [
    {
      "type": "Rectangle",
      "operation": "add",
      "name": "Flangia superiore",
      "params": { "b": 200, "h": 15, "x0": 0, "y0": 185 }
    },
    {
      "type": "Circle",
      "operation": "subtract",
      "name": "Foro centrale",
      "params": { "r": 20, "xc": 100, "yc": 100 }
    }
  ],
  "notes": "osservazioni opzionali"
}

Tipi di forme disponibili e parametri richiesti:
  Rectangle   : b, h, x0, y0       (angolo inferiore sinistro)
  Circle      : r, xc, yc
  HollowCircle: R (est.), r (int.), xc, yc
  Triangle    : b, h, x0, y0       (vertice retto in basso a sinistra)
  Semicircle  : r, x0, y0          (centro del diametro piatto)
  Polygon     : vertices            (lista [[x,y], ...] in ordine antiorario)

Regole importanti:
  - operation = "add"      per materiale pieno
  - operation = "subtract" per fori o cavità
  - Origine del sistema di riferimento: in basso a sinistra
  - Se nell'immagine ci sono quote esplicite, usale ESATTAMENTE
  - Se non ci sono quote, stima le dimensioni in pixel relativi mantenendo
    le proporzioni corrette
  - Suddividi sezioni complesse nella somma di forme semplici
"""


class ClaudeVisionAnalyzer:
    """
    Interpreta disegni tecnici, schizzi e foto di sezioni trasversali
    utilizzando Claude Vision API.

    Riconosce forme geometriche, quote e simboli di foro/cavità,
    restituendo automaticamente una CrossSection pronta per il calcolo.

    Richiede:
      - pip install anthropic
      - variabile d'ambiente ANTHROPIC_API_KEY  oppure  api_key=<chiave>
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-4-6"):
        try:
            import anthropic as _anthropic
            self._anthropic = _anthropic
        except ImportError:
            raise ImportError("Installare anthropic:\n  pip install anthropic")

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "API key non trovata.\n"
                "Impostare la variabile d'ambiente ANTHROPIC_API_KEY\n"
                "oppure passare api_key=<chiave> al costruttore."
            )
        self.client = self._anthropic.Anthropic(api_key=key)
        self.model = model

    def analyze(self, image_path: str) -> CrossSection:
        """
        Invia l'immagine a Claude e costruisce la CrossSection dal risultato.

        Args:
            image_path: percorso all'immagine (jpg, png, gif, webp)

        Returns:
            CrossSection pronta per il calcolo
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Immagine non trovata: {image_path}")

        _media_types = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = _media_types.get(path.suffix.lower(), "image/png")

        with open(path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("utf-8")

        print("  ⏳ Invio immagine a Claude Vision...")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=_CLAUDE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Analizza questa sezione trasversale ed estrai "
                                "la geometria come JSON strutturato."
                            ),
                        },
                    ],
                }
            ],
        )

        raw = response.content[0].text.strip()

        # Rimuovi eventuale blocco markdown ```json … ```
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    data = json.loads(part)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                raise ValueError(f"Impossibile estrarre JSON dalla risposta:\n{raw}")
        else:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Risposta Claude non è JSON valido:\n{raw}\n{e}")

        print(f"  ✓ Claude ha identificato {len(data.get('shapes', []))} forme.")
        if data.get("notes"):
            print(f"  📝 Note: {data['notes']}")
        if data.get("unit"):
            print(f"  📐 Unità: {data['unit']}")

        return self._build_section(data)

    def _build_section(self, data: dict) -> CrossSection:
        """Costruisce la CrossSection dal dizionario JSON restituito da Claude."""
        section = CrossSection(data.get("section_name", "Sezione da immagine"))

        for sd in data.get("shapes", []):
            op    = sd.get("operation", "add")
            name  = sd.get("name", sd.get("type", "Shape"))
            stype = sd.get("type", "")
            params = sd.get("params", {})

            # Gestione Polygon: vertices può stare in params o nel dict root
            if stype == "Polygon":
                verts = params.get("vertices") or sd.get("vertices", [])
                params = {"vertices": [tuple(v) for v in verts]}

            d = {"type": stype, "name": name, "sign": 1, **params}
            try:
                shape = _shape_from_dict(d)
                if op == "subtract":
                    section.subtract(shape)
                else:
                    section.add(shape)
            except (ValueError, KeyError, TypeError) as e:
                print(f"  ⚠  Forma '{name}' ({stype}) ignorata: {e}")

        if not section.shapes:
            raise ValueError("Nessuna forma valida estratta dalla risposta AI.")
        return section


# ─── 3. Visualizzatore della sezione ────────────────────────────────────────

def visualize_section(section: CrossSection, show_centroid: bool = True) -> None:
    """
    Disegna le forme della CrossSection con matplotlib.
    Mostra il baricentro, gli assi baricentrici e il report delle proprietà.

    Args:
        section:        la sezione da visualizzare
        show_centroid:  se True, mostra il punto G e gli assi
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patheffects as pe

    if not section.shapes:
        print("  Sezione vuota.")
        return

    try:
        props = section.calculate()
    except ValueError as e:
        print(f"  ✗ {e}")
        return

    fig, (ax_geo, ax_info) = plt.subplots(
        1, 2, figsize=(13, 7),
        gridspec_kw={"width_ratios": [3, 2]}
    )
    fig.suptitle(f'Sezione: "{section.name}"', fontsize=13, fontweight="bold")

    # ── Disegno delle forme ──────────────────────────────────────────────
    ADD_COLOR = "#90CAF9"      # blu chiaro per materiale pieno
    SUB_COLOR = "#FFFFFF"      # bianco per fori
    EDGE_ADD  = "#1565C0"
    EDGE_SUB  = "#B71C1C"

    all_xs, all_ys = [], []

    for shape in section.shapes:
        color = ADD_COLOR if shape.sign == 1 else SUB_COLOR
        edge  = EDGE_ADD  if shape.sign == 1 else EDGE_SUB
        lw    = 1.5       if shape.sign == 1 else 1.0
        zord  = 2         if shape.sign == 1 else 3

        if isinstance(shape, Polygon):
            xs = [v[0] for v in shape.vertices]
            ys = [v[1] for v in shape.vertices]
            ax_geo.fill(xs, ys, color=color, zorder=zord)
            ax_geo.plot(xs + [xs[0]], ys + [ys[0]], color=edge, lw=lw, zorder=zord+1)
            all_xs.extend(xs)
            all_ys.extend(ys)

        elif isinstance(shape, Rectangle):
            from matplotlib.patches import Rectangle as MplRect
            x0, y0, b, h = shape.x0, shape.y0, shape.b, shape.h
            rect = MplRect((x0, y0), b, h,
                           facecolor=color, edgecolor=edge, lw=lw, zorder=zord)
            ax_geo.add_patch(rect)
            all_xs += [x0, x0+b]; all_ys += [y0, y0+h]

        elif isinstance(shape, (Circle, HollowCircle)):
            from matplotlib.patches import Circle as MplCircle, Wedge
            xc, yc = shape.centroid()
            if isinstance(shape, HollowCircle):
                outer = MplCircle((xc, yc), shape.R,
                                  facecolor=color, edgecolor=edge, lw=lw, zorder=zord)
                inner = MplCircle((xc, yc), shape.r,
                                  facecolor=SUB_COLOR, edgecolor=EDGE_SUB, lw=1, zorder=zord+1)
                ax_geo.add_patch(outer)
                ax_geo.add_patch(inner)
                all_xs += [xc-shape.R, xc+shape.R]
                all_ys += [yc-shape.R, yc+shape.R]
            else:
                circ = MplCircle((xc, yc), shape.r,
                                 facecolor=color, edgecolor=edge, lw=lw, zorder=zord)
                ax_geo.add_patch(circ)
                all_xs += [xc-shape.r, xc+shape.r]
                all_ys += [yc-shape.r, yc+shape.r]

        elif isinstance(shape, Triangle):
            x0, y0, b, h = shape.x0, shape.y0, shape.b, shape.h
            xs = [x0, x0+b, x0]; ys = [y0, y0, y0+h]
            ax_geo.fill(xs, ys, color=color, zorder=zord)
            ax_geo.plot(xs+[xs[0]], ys+[ys[0]], color=edge, lw=lw, zorder=zord+1)
            all_xs.extend(xs); all_ys.extend(ys)

        elif isinstance(shape, Semicircle):
            from matplotlib.patches import Wedge
            xc_s, yc_s = shape.x0, shape.y0
            wedge = Wedge((xc_s, yc_s), shape.r, 0, 180,
                          facecolor=color, edgecolor=edge, lw=lw, zorder=zord)
            ax_geo.add_patch(wedge)
            all_xs += [xc_s-shape.r, xc_s+shape.r]
            all_ys += [yc_s, yc_s+shape.r]

    # ── Baricentro e assi baricentrici ───────────────────────────────────
    if show_centroid and all_xs:
        margin = max(
            (max(all_xs) - min(all_xs)) * 0.1,
            (max(all_ys) - min(all_ys)) * 0.1,
            1.0
        )
        arm = margin * 1.2

        ax_geo.axhline(props.yc, color="#E53935", lw=1.2,
                       ls="--", alpha=0.7, zorder=5, label="Asse Gx")
        ax_geo.axvline(props.xc, color="#43A047", lw=1.2,
                       ls="--", alpha=0.7, zorder=5, label="Asse Gy")
        ax_geo.plot(props.xc, props.yc, "r*", ms=12, zorder=6, label="Baricentro G")
        ax_geo.annotate(
            f" G({props.xc:.2f}, {props.yc:.2f})",
            xy=(props.xc, props.yc),
            xytext=(props.xc + arm * 0.3, props.yc + arm * 0.3),
            fontsize=8, color="#B71C1C",
            arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=0.8),
            zorder=7
        )

    # ── Messa in scala degli assi ────────────────────────────────────────
    if all_xs and all_ys:
        dx = max(all_xs) - min(all_xs) or 1
        dy = max(all_ys) - min(all_ys) or 1
        pad = max(dx, dy) * 0.12
        ax_geo.set_xlim(min(all_xs) - pad, max(all_xs) + pad)
        ax_geo.set_ylim(min(all_ys) - pad, max(all_ys) + pad)

    ax_geo.set_aspect("equal")
    ax_geo.set_xlabel("x")
    ax_geo.set_ylabel("y")
    ax_geo.legend(fontsize=8, loc="upper right")
    ax_geo.grid(True, alpha=0.3)

    # ── Pannello testo (proprietà) ───────────────────────────────────────
    ax_info.axis("off")
    lines = [
        ("PROPRIETÀ DELLA SEZIONE", None, True),
        ("", None, False),
        ("GEOMETRIA", "header", False),
        (f"Area  A", f"{props.area:.4g}", False),
        (f"Baricentro  xc", f"{props.xc:.4g}", False),
        (f"Baricentro  yc", f"{props.yc:.4g}", False),
        ("", None, False),
        ("MOMENTI STATICI", "header", False),
        ("Sx  =  ∫y dA", f"{props.Sx:.4g}", False),
        ("Sy  =  ∫x dA", f"{props.Sy:.4g}", False),
        ("", None, False),
        ("MOM. INERZIA  (origine)", "header", False),
        ("Ix  =  ∫y² dA", f"{props.Ix:.4g}", False),
        ("Iy  =  ∫x² dA", f"{props.Iy:.4g}", False),
        ("Ixy = ∫xy dA", f"{props.Ixy:.4g}", False),
        ("", None, False),
        ("MOM. INERZIA  (baricentro)", "header", False),
        ("Ixc", f"{props.Ix_c:.4g}", False),
        ("Iyc", f"{props.Iy_c:.4g}", False),
        ("Ixyc", f"{props.Ixy_c:.4g}", False),
        ("", None, False),
        ("RAGGI DI GIRAZIONE", "header", False),
        ("ix = √(Ixc/A)", f"{props.ix:.4g}", False),
        ("iy = √(Iyc/A)", f"{props.iy:.4g}", False),
    ]

    y_pos = 0.97
    for label, value, bold in lines:
        if label == "":
            y_pos -= 0.025
            continue
        if value is None and bold:
            ax_info.text(0.05, y_pos, label, transform=ax_info.transAxes,
                         fontsize=11, fontweight="bold", color="#1A237E")
        elif value == "header":
            ax_info.text(0.05, y_pos, label, transform=ax_info.transAxes,
                         fontsize=9, fontweight="bold", color="#37474F",
                         bbox=dict(boxstyle="round,pad=0.2", fc="#ECEFF1", ec="none"))
        else:
            ax_info.text(0.05, y_pos, label, transform=ax_info.transAxes,
                         fontsize=8.5, color="#212121")
            ax_info.text(0.72, y_pos, value, transform=ax_info.transAxes,
                         fontsize=8.5, color="#1565C0", ha="right",
                         fontfamily="monospace")
        y_pos -= 0.038

    plt.tight_layout()
    plt.show()
