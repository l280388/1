"""
Geometry Mass Calculator
========================
Calcola area, momenti statici e momenti d'inerzia di sezioni trasversali complesse.

Forme supportate:
  - Rettangolo
  - Cerchio
  - Anello circolare (sezione cava)
  - Triangolo rettangolo
  - Semicerchio
  - Poligono generico (vertici arbitrari)

Le forme possono essere sommate o sottratte per costruire sezioni complesse.
"""

import math
import json
from dataclasses import dataclass
from typing import List, Tuple
from abc import ABC, abstractmethod


# ─── Risultati ───────────────────────────────────────────────────────────────

@dataclass
class SectionProperties:
    """Proprietà geometriche della sezione trasversale."""
    area: float          # Area totale A
    Sx: float            # Momento statico rispetto all'asse X: Sx = ∫y dA
    Sy: float            # Momento statico rispetto all'asse Y: Sy = ∫x dA
    xc: float            # Coordinata X del baricentro G
    yc: float            # Coordinata Y del baricentro G
    Ix: float            # Momento d'inerzia rispetto all'asse X (origine): Ix = ∫y² dA
    Iy: float            # Momento d'inerzia rispetto all'asse Y (origine): Iy = ∫x² dA
    Ixy: float           # Prodotto d'inerzia (origine): Ixy = ∫xy dA
    Ix_c: float          # Momento d'inerzia rispetto all'asse baricentrico Gx
    Iy_c: float          # Momento d'inerzia rispetto all'asse baricentrico Gy
    Ixy_c: float         # Prodotto d'inerzia baricentrico
    ix: float            # Raggio di girazione: ix = √(Ix_c / A)
    iy: float            # Raggio di girazione: iy = √(Iy_c / A)

    def summary(self, title: str = "PROPRIETÀ DELLA SEZIONE") -> str:
        """Restituisce un report formattato."""
        w = 54
        sep = "═" * w
        thin = "─" * w

        def row(label, symbol, value, unit=""):
            return f"  {label:<32} {symbol:<5} = {value:>14.4f}  {unit}"

        lines = [
            sep,
            f"  {title}".center(w),
            sep,
            "",
            "  GEOMETRIA",
            thin,
            row("Area totale", "A", self.area, ""),
            row("Baricentro X", "xc", self.xc, ""),
            row("Baricentro Y", "yc", self.yc, ""),
            "",
            "  MOMENTI STATICI (rispetto all'origine)",
            thin,
            row("Momento statico / asse X", "Sx", self.Sx, ""),
            row("Momento statico / asse Y", "Sy", self.Sy, ""),
            "",
            "  MOMENTI D'INERZIA (rispetto all'origine)",
            thin,
            row("Momento d'inerzia / asse X", "Ix", self.Ix, ""),
            row("Momento d'inerzia / asse Y", "Iy", self.Iy, ""),
            row("Prodotto d'inerzia", "Ixy", self.Ixy, ""),
            "",
            "  MOMENTI D'INERZIA BARICENTRICI",
            thin,
            row("Mom. d'inerzia asse Gx", "Ixc", self.Ix_c, ""),
            row("Mom. d'inerzia asse Gy", "Iyc", self.Iy_c, ""),
            row("Prodotto d'inerzia baricentrico", "Ixyc", self.Ixy_c, ""),
            "",
            "  RAGGI DI GIRAZIONE",
            thin,
            row("Raggio di girazione x", "ix", self.ix, ""),
            row("Raggio di girazione y", "iy", self.iy, ""),
            "",
            sep,
        ]
        return "\n".join(lines)


# ─── Classe base Shape ────────────────────────────────────────────────────────

class Shape(ABC):
    """Classe base per tutte le forme geometriche."""

    def __init__(self, name: str, sign: int = 1):
        self.name = name
        self.sign = sign  # +1 → addizione, -1 → sottrazione

    @abstractmethod
    def area(self) -> float:
        """Area della forma."""

    @abstractmethod
    def centroid(self) -> Tuple[float, float]:
        """Coordinate (xc, yc) del baricentro."""

    @abstractmethod
    def Ix_centroid(self) -> float:
        """Momento d'inerzia rispetto all'asse baricentrico X."""

    @abstractmethod
    def Iy_centroid(self) -> float:
        """Momento d'inerzia rispetto all'asse baricentrico Y."""

    @abstractmethod
    def Ixy_centroid(self) -> float:
        """Prodotto d'inerzia rispetto agli assi baricentrici."""

    # ── Teorema di Steiner (trasposizione agli assi di riferimento) ──────────

    def Ix_axis(self) -> float:
        """Ix rispetto all'asse X di riferimento (y=0)."""
        _, yc = self.centroid()
        return self.Ix_centroid() + self.area() * yc ** 2

    def Iy_axis(self) -> float:
        """Iy rispetto all'asse Y di riferimento (x=0)."""
        xc, _ = self.centroid()
        return self.Iy_centroid() + self.area() * xc ** 2

    def Ixy_axis(self) -> float:
        """Ixy rispetto agli assi di riferimento."""
        xc, yc = self.centroid()
        return self.Ixy_centroid() + self.area() * xc * yc

    @abstractmethod
    def _params(self) -> dict:
        """Parametri per la serializzazione JSON."""

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "sign": self.sign,
            **self._params(),
        }

    def __str__(self) -> str:
        op = "+" if self.sign == 1 else "-"
        xc, yc = self.centroid()
        return (
            f"  [{op}] {self.name:<22}  A={self.area():>12.4f}   "
            f"G=({xc:.4f}, {yc:.4f})"
        )


# ─── Forme primitive ─────────────────────────────────────────────────────────

class Rectangle(Shape):
    """
    Rettangolo di base b e altezza h.
    Parametri di posizione: angolo inferiore sinistro (x0, y0).
    Baricentro in (x0 + b/2, y0 + h/2).
    """

    def __init__(self, b: float, h: float,
                 x0: float = 0.0, y0: float = 0.0,
                 name: str = "Rettangolo", sign: int = 1):
        super().__init__(name, sign)
        if b <= 0 or h <= 0:
            raise ValueError("Base e altezza devono essere positive.")
        self.b = b
        self.h = h
        self.x0 = x0
        self.y0 = y0

    def area(self) -> float:
        return self.b * self.h

    def centroid(self) -> Tuple[float, float]:
        return (self.x0 + self.b / 2, self.y0 + self.h / 2)

    def Ix_centroid(self) -> float:
        return self.b * self.h ** 3 / 12

    def Iy_centroid(self) -> float:
        return self.h * self.b ** 3 / 12

    def Ixy_centroid(self) -> float:
        return 0.0

    def _params(self) -> dict:
        return {"b": self.b, "h": self.h, "x0": self.x0, "y0": self.y0}

    def __str__(self) -> str:
        op = "+" if self.sign == 1 else "-"
        xc, yc = self.centroid()
        return (
            f"  [{op}] {self.name:<22}  b={self.b:.4f} h={self.h:.4f}  "
            f"A={self.area():>12.4f}   G=({xc:.4f}, {yc:.4f})"
        )


class Circle(Shape):
    """
    Cerchio di raggio r con centro in (xc, yc).
    """

    def __init__(self, r: float,
                 xc: float = 0.0, yc: float = 0.0,
                 name: str = "Cerchio", sign: int = 1):
        super().__init__(name, sign)
        if r <= 0:
            raise ValueError("Il raggio deve essere positivo.")
        self.r = r
        self._xc = xc
        self._yc = yc

    def area(self) -> float:
        return math.pi * self.r ** 2

    def centroid(self) -> Tuple[float, float]:
        return (self._xc, self._yc)

    def Ix_centroid(self) -> float:
        return math.pi * self.r ** 4 / 4

    def Iy_centroid(self) -> float:
        return math.pi * self.r ** 4 / 4

    def Ixy_centroid(self) -> float:
        return 0.0

    def _params(self) -> dict:
        return {"r": self.r, "xc": self._xc, "yc": self._yc}

    def __str__(self) -> str:
        op = "+" if self.sign == 1 else "-"
        return (
            f"  [{op}] {self.name:<22}  r={self.r:.4f}              "
            f"A={self.area():>12.4f}   G=({self._xc:.4f}, {self._yc:.4f})"
        )


class HollowCircle(Shape):
    """
    Anello circolare (sezione cava).
    Raggio esterno R, raggio interno r, centro in (xc, yc).
    """

    def __init__(self, R: float, r: float,
                 xc: float = 0.0, yc: float = 0.0,
                 name: str = "Anello", sign: int = 1):
        super().__init__(name, sign)
        if r >= R:
            raise ValueError("Il raggio interno deve essere < raggio esterno.")
        if r <= 0 or R <= 0:
            raise ValueError("I raggi devono essere positivi.")
        self.R = R
        self.r = r
        self._xc = xc
        self._yc = yc

    def area(self) -> float:
        return math.pi * (self.R ** 2 - self.r ** 2)

    def centroid(self) -> Tuple[float, float]:
        return (self._xc, self._yc)

    def Ix_centroid(self) -> float:
        return math.pi * (self.R ** 4 - self.r ** 4) / 4

    def Iy_centroid(self) -> float:
        return math.pi * (self.R ** 4 - self.r ** 4) / 4

    def Ixy_centroid(self) -> float:
        return 0.0

    def _params(self) -> dict:
        return {"R": self.R, "r": self.r, "xc": self._xc, "yc": self._yc}

    def __str__(self) -> str:
        op = "+" if self.sign == 1 else "-"
        return (
            f"  [{op}] {self.name:<22}  R={self.R:.4f} r={self.r:.4f}  "
            f"A={self.area():>12.4f}   G=({self._xc:.4f}, {self._yc:.4f})"
        )


class Triangle(Shape):
    """
    Triangolo rettangolo con base b e altezza h.
    Il vertice retto è in (x0, y0); la base va verso destra (asse +x),
    l'altezza verso l'alto (asse +y).
    Baricentro in (x0 + b/3, y0 + h/3).

    Momenti d'inerzia baricentrici:
      Ix_c = b·h³/36
      Iy_c = h·b³/36
      Ixy_c = −b²·h²/72  (per questa orientazione)
    """

    def __init__(self, b: float, h: float,
                 x0: float = 0.0, y0: float = 0.0,
                 name: str = "Triangolo", sign: int = 1):
        super().__init__(name, sign)
        if b <= 0 or h <= 0:
            raise ValueError("Base e altezza devono essere positive.")
        self.b = b
        self.h = h
        self.x0 = x0
        self.y0 = y0

    def area(self) -> float:
        return self.b * self.h / 2

    def centroid(self) -> Tuple[float, float]:
        return (self.x0 + self.b / 3, self.y0 + self.h / 3)

    def Ix_centroid(self) -> float:
        return self.b * self.h ** 3 / 36

    def Iy_centroid(self) -> float:
        return self.h * self.b ** 3 / 36

    def Ixy_centroid(self) -> float:
        # Triangolo rettangolo con cateti lungo +x e +y
        return -(self.b ** 2 * self.h ** 2) / 72

    def _params(self) -> dict:
        return {"b": self.b, "h": self.h, "x0": self.x0, "y0": self.y0}

    def __str__(self) -> str:
        op = "+" if self.sign == 1 else "-"
        xc, yc = self.centroid()
        return (
            f"  [{op}] {self.name:<22}  b={self.b:.4f} h={self.h:.4f}  "
            f"A={self.area():>12.4f}   G=({xc:.4f}, {yc:.4f})"
        )


class Semicircle(Shape):
    """
    Semicerchio con raggio r, lato piatto verso il basso.
    Il centro del diametro piatto è in (x0, y0);
    il semicerchio si estende verso l'alto.
    Baricentro in (x0, y0 + 4r/(3π)).

    Momenti d'inerzia baricentrici:
      Ix_c = (π/8 − 8/(9π))·r⁴  ≈ 0.10976·r⁴
      Iy_c = π·r⁴/8
      Ixy_c = 0  (simmetria rispetto all'asse verticale)
    """

    def __init__(self, r: float,
                 x0: float = 0.0, y0: float = 0.0,
                 name: str = "Semicerchio", sign: int = 1):
        super().__init__(name, sign)
        if r <= 0:
            raise ValueError("Il raggio deve essere positivo.")
        self.r = r
        self.x0 = x0
        self.y0 = y0

    def area(self) -> float:
        return math.pi * self.r ** 2 / 2

    def centroid(self) -> Tuple[float, float]:
        return (self.x0, self.y0 + 4 * self.r / (3 * math.pi))

    def Ix_centroid(self) -> float:
        return (math.pi / 8 - 8 / (9 * math.pi)) * self.r ** 4

    def Iy_centroid(self) -> float:
        return math.pi * self.r ** 4 / 8

    def Ixy_centroid(self) -> float:
        return 0.0

    def _params(self) -> dict:
        return {"r": self.r, "x0": self.x0, "y0": self.y0}

    def __str__(self) -> str:
        op = "+" if self.sign == 1 else "-"
        xc, yc = self.centroid()
        return (
            f"  [{op}] {self.name:<22}  r={self.r:.4f}              "
            f"A={self.area():>12.4f}   G=({xc:.4f}, {yc:.4f})"
        )


class Polygon(Shape):
    """
    Poligono generico definito da una lista di vertici (x, y).
    I vertici devono essere forniti in ordine antiorario (CCW) per ottenere
    un'area positiva; l'ordine orario dà area negativa.

    Calcolo tramite formule di Green:
      A    = (1/2) Σ (xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)
      xc   = (1/6A) Σ (xᵢ + xᵢ₊₁)(xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)
      yc   = (1/6A) Σ (yᵢ + yᵢ₊₁)(xᵢ·yᵢ₊₁ − xᵢ₊₁·yᵢ)
      Ix   = (1/12)Σ (yᵢ² + yᵢyᵢ₊₁ + yᵢ₊₁²)(xᵢyᵢ₊₁ − xᵢ₊₁yᵢ)
      Iy   = (1/12)Σ (xᵢ² + xᵢxᵢ₊₁ + xᵢ₊₁²)(xᵢyᵢ₊₁ − xᵢ₊₁yᵢ)
      Ixy  = (1/24)Σ (xᵢyᵢ₊₁ + 2xᵢyᵢ + 2xᵢ₊₁yᵢ₊₁ + xᵢ₊₁yᵢ)(xᵢyᵢ₊₁ − xᵢ₊₁yᵢ)
    """

    def __init__(self, vertices: List[Tuple[float, float]],
                 name: str = "Poligono", sign: int = 1):
        super().__init__(name, sign)
        if len(vertices) < 3:
            raise ValueError("Il poligono deve avere almeno 3 vertici.")
        self.vertices = list(vertices)
        self._compute()

    def _compute(self):
        verts = self.vertices
        n = len(verts)
        A = 0.0
        Cx = 0.0
        Cy = 0.0
        Ix = 0.0
        Iy = 0.0
        Ixy = 0.0

        for i in range(n):
            xi, yi = verts[i]
            xj, yj = verts[(i + 1) % n]
            d = xi * yj - xj * yi  # termine della formula dell'area

            A += d
            Cx += (xi + xj) * d
            Cy += (yi + yj) * d
            Ix += (yi ** 2 + yi * yj + yj ** 2) * d
            Iy += (xi ** 2 + xi * xj + xj ** 2) * d
            Ixy += (xi * yj + 2 * xi * yi + 2 * xj * yj + xj * yi) * d

        self._area = A / 2
        if abs(self._area) < 1e-12:
            raise ValueError("Area del poligono nulla o degenere.")

        self._xc = Cx / (6 * self._area)
        self._yc = Cy / (6 * self._area)
        self._Ix_axis = Ix / 12   # rispetto all'asse X di riferimento
        self._Iy_axis = Iy / 12
        self._Ixy_axis = Ixy / 24

        # Momenti baricentrici (Steiner inverso)
        self._Ix_c = self._Ix_axis - self._area * self._yc ** 2
        self._Iy_c = self._Iy_axis - self._area * self._xc ** 2
        self._Ixy_c = self._Ixy_axis - self._area * self._xc * self._yc

    def area(self) -> float:
        return self._area

    def centroid(self) -> Tuple[float, float]:
        return (self._xc, self._yc)

    def Ix_centroid(self) -> float:
        return self._Ix_c

    def Iy_centroid(self) -> float:
        return self._Iy_c

    def Ixy_centroid(self) -> float:
        return self._Ixy_c

    # Override: per il poligono i momenti sull'asse di riferimento sono già
    # calcolati direttamente (non via Steiner), così si evita doppio calcolo.
    def Ix_axis(self) -> float:
        return self._Ix_axis

    def Iy_axis(self) -> float:
        return self._Iy_axis

    def Ixy_axis(self) -> float:
        return self._Ixy_axis

    def _params(self) -> dict:
        return {"vertices": self.vertices}

    def __str__(self) -> str:
        op = "+" if self.sign == 1 else "-"
        n = len(self.vertices)
        return (
            f"  [{op}] {self.name:<22}  ({n} vertici)           "
            f"A={self.area():>12.4f}   G=({self._xc:.4f}, {self._yc:.4f})"
        )


# ─── Sezione composta ─────────────────────────────────────────────────────────

class CrossSection:
    """
    Sezione trasversale composta da più forme geometriche.
    Ogni forma può essere aggiunta (+) o sottratta (−, per i fori).
    """

    def __init__(self, name: str = "Sezione"):
        self.name = name
        self.shapes: List[Shape] = []

    # ── Gestione forme ───────────────────────────────────────────────────────

    def add(self, shape: Shape) -> None:
        """Aggiunge una forma alla sezione (somma)."""
        shape.sign = 1
        self.shapes.append(shape)

    def subtract(self, shape: Shape) -> None:
        """Sottrae una forma dalla sezione (foro)."""
        shape.sign = -1
        self.shapes.append(shape)

    def remove(self, index: int) -> None:
        """Rimuove la forma all'indice specificato."""
        if not (0 <= index < len(self.shapes)):
            raise IndexError("Indice fuori range.")
        self.shapes.pop(index)

    def clear(self) -> None:
        """Rimuove tutte le forme."""
        self.shapes.clear()

    # ── Calcolo ─────────────────────────────────────────────────────────────

    def calculate(self) -> SectionProperties:
        """Calcola e restituisce le proprietà della sezione composta."""
        if not self.shapes:
            raise ValueError("Nessuna forma nella sezione.")

        total_A = 0.0
        total_Sx = 0.0   # Σ ±A·yc
        total_Sy = 0.0   # Σ ±A·xc
        total_Ix = 0.0   # Σ ±Ix (rispetto all'origine)
        total_Iy = 0.0
        total_Ixy = 0.0

        for shape in self.shapes:
            s = shape.sign
            A = shape.area()
            xc, yc = shape.centroid()
            total_A += s * A
            total_Sx += s * A * yc
            total_Sy += s * A * xc
            total_Ix += s * shape.Ix_axis()
            total_Iy += s * shape.Iy_axis()
            total_Ixy += s * shape.Ixy_axis()

        if abs(total_A) < 1e-12:
            raise ValueError("Area totale nulla: verificare la geometria.")

        xc = total_Sy / total_A
        yc = total_Sx / total_A

        # Momenti d'inerzia baricentrici (Steiner inverso)
        Ix_c = total_Ix - total_A * yc ** 2
        Iy_c = total_Iy - total_A * xc ** 2
        Ixy_c = total_Ixy - total_A * xc * yc

        ix = math.sqrt(abs(Ix_c / total_A))
        iy = math.sqrt(abs(Iy_c / total_A))

        return SectionProperties(
            area=total_A,
            Sx=total_Sx,
            Sy=total_Sy,
            xc=xc,
            yc=yc,
            Ix=total_Ix,
            Iy=total_Iy,
            Ixy=total_Ixy,
            Ix_c=Ix_c,
            Iy_c=Iy_c,
            Ixy_c=Ixy_c,
            ix=ix,
            iy=iy,
        )

    # ── Serializzazione ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shapes": [s.to_dict() for s in self.shapes],
        }

    def save(self, filename: str) -> None:
        """Salva la sezione su file JSON."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filename: str) -> "CrossSection":
        """Carica una sezione da file JSON."""
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        section = cls(data.get("name", "Sezione"))
        for sd in data.get("shapes", []):
            shape = _shape_from_dict(sd)
            section.shapes.append(shape)
        return section

    def __str__(self) -> str:
        if not self.shapes:
            return f'Sezione "{self.name}" — vuota'
        lines = [f'Sezione "{self.name}" — {len(self.shapes)} forma/e:']
        for i, s in enumerate(self.shapes):
            lines.append(f"  {i:2d}. {str(s).strip()}")
        return "\n".join(lines)


# ─── Utility ─────────────────────────────────────────────────────────────────

_SHAPE_REGISTRY = {
    "Rectangle": Rectangle,
    "Circle": Circle,
    "HollowCircle": HollowCircle,
    "Triangle": Triangle,
    "Semicircle": Semicircle,
    "Polygon": Polygon,
}


def _shape_from_dict(d: dict) -> Shape:
    """Ricostruisce una Shape da un dizionario JSON."""
    t = d["type"]
    if t not in _SHAPE_REGISTRY:
        raise ValueError(f"Tipo di forma sconosciuto: {t}")
    cls = _SHAPE_REGISTRY[t]
    name = d.get("name", t)
    sign = d.get("sign", 1)
    params = {k: v for k, v in d.items() if k not in ("type", "name", "sign")}
    if t == "Polygon":
        params["vertices"] = [tuple(v) for v in params["vertices"]]
    shape = cls(**params, name=name)
    shape.sign = sign
    return shape
