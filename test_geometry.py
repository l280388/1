"""
Unit tests per geometry_calculator.py
Verificano i valori analitici noti per ogni forma.
"""
import math
import pytest
from geometry_calculator import (
    Rectangle, Circle, HollowCircle, Triangle, Semicircle, Polygon,
    CrossSection, SectionProperties
)

EPS = 1e-8


# ─── Rectangle ───────────────────────────────────────────────────────────────

class TestRectangle:
    def test_area(self):
        r = Rectangle(4, 6)
        assert abs(r.area() - 24) < EPS

    def test_centroid(self):
        r = Rectangle(4, 6, x0=1, y0=2)
        xc, yc = r.centroid()
        assert abs(xc - 3.0) < EPS
        assert abs(yc - 5.0) < EPS

    def test_Ix_centroid(self):
        # b·h³/12 = 4·6³/12 = 72
        r = Rectangle(4, 6)
        assert abs(r.Ix_centroid() - 72) < EPS

    def test_Iy_centroid(self):
        # h·b³/12 = 6·4³/12 = 32
        r = Rectangle(4, 6)
        assert abs(r.Iy_centroid() - 32) < EPS

    def test_Ixy_centroid_zero(self):
        r = Rectangle(4, 6, 1, 2)
        assert abs(r.Ixy_centroid()) < EPS

    def test_Ix_axis(self):
        # Ix = b·h³/12 + A·yc² = 72 + 24·5² = 72+600 = 672
        r = Rectangle(4, 6, x0=1, y0=2)
        assert abs(r.Ix_axis() - 672) < EPS

    def test_Iy_axis(self):
        # Iy = h·b³/12 + A·xc² = 32 + 24·9 = 32+216 = 248
        r = Rectangle(4, 6, x0=1, y0=2)
        assert abs(r.Iy_axis() - 248) < EPS

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            Rectangle(-1, 5)

    def test_serialization(self):
        r = Rectangle(3, 7, 1, 2, name="Test")
        d = r.to_dict()
        assert d["type"] == "Rectangle"
        assert d["b"] == 3
        assert d["name"] == "Test"


# ─── Circle ──────────────────────────────────────────────────────────────────

class TestCircle:
    def test_area(self):
        c = Circle(5)
        assert abs(c.area() - math.pi * 25) < EPS

    def test_centroid(self):
        c = Circle(5, xc=3, yc=4)
        assert c.centroid() == (3, 4)

    def test_Ix_centroid(self):
        # π·r⁴/4 = π·625/4
        c = Circle(5)
        assert abs(c.Ix_centroid() - math.pi * 625 / 4) < EPS

    def test_Ix_axis(self):
        # Ix = π·r⁴/4 + A·yc²  → yc=4, A=25π
        c = Circle(5, xc=3, yc=4)
        expected = math.pi * 625 / 4 + math.pi * 25 * 16
        assert abs(c.Ix_axis() - expected) < EPS

    def test_invalid_radius(self):
        with pytest.raises(ValueError):
            Circle(0)


# ─── HollowCircle ────────────────────────────────────────────────────────────

class TestHollowCircle:
    def test_area(self):
        h = HollowCircle(5, 3)
        assert abs(h.area() - math.pi * (25 - 9)) < EPS

    def test_Ix_centroid(self):
        # π(R⁴-r⁴)/4
        h = HollowCircle(5, 3)
        assert abs(h.Ix_centroid() - math.pi * (625 - 81) / 4) < EPS

    def test_invalid(self):
        with pytest.raises(ValueError):
            HollowCircle(3, 5)  # r > R


# ─── Triangle ────────────────────────────────────────────────────────────────

class TestTriangle:
    def test_area(self):
        t = Triangle(6, 4)
        assert abs(t.area() - 12) < EPS

    def test_centroid(self):
        # vertice retto a (0,0): G a (6/3, 4/3) = (2, 4/3)
        t = Triangle(6, 4)
        xc, yc = t.centroid()
        assert abs(xc - 2.0) < EPS
        assert abs(yc - 4 / 3) < EPS

    def test_centroid_offset(self):
        t = Triangle(6, 4, x0=1, y0=2)
        xc, yc = t.centroid()
        assert abs(xc - 3.0) < EPS
        assert abs(yc - (2 + 4 / 3)) < 1e-7

    def test_Ix_centroid(self):
        # b·h³/36 = 6·64/36 = 384/36
        t = Triangle(6, 4)
        assert abs(t.Ix_centroid() - 6 * 64 / 36) < EPS

    def test_Ixy_centroid(self):
        # −b²h²/72 = −36·16/72 = −8
        t = Triangle(6, 4)
        assert abs(t.Ixy_centroid() - (-8)) < EPS


# ─── Semicircle ──────────────────────────────────────────────────────────────

class TestSemicircle:
    def test_area(self):
        s = Semicircle(5)
        assert abs(s.area() - math.pi * 25 / 2) < EPS

    def test_centroid(self):
        s = Semicircle(5, x0=0, y0=0)
        xc, yc = s.centroid()
        assert abs(xc) < EPS
        assert abs(yc - 4 * 5 / (3 * math.pi)) < EPS

    def test_Iy_centroid(self):
        # π·r⁴/8
        s = Semicircle(5)
        assert abs(s.Iy_centroid() - math.pi * 625 / 8) < EPS

    def test_Ix_centroid(self):
        # (π/8 − 8/(9π))·r⁴
        s = Semicircle(5)
        expected = (math.pi / 8 - 8 / (9 * math.pi)) * 625
        assert abs(s.Ix_centroid() - expected) < EPS

    def test_steiner_check(self):
        """Ix rispetto all'asse piatto == Ix_c + A·yc²."""
        s = Semicircle(5)
        A = s.area()
        _, yc = s.centroid()
        Ix_flat = s.Ix_centroid() + A * yc ** 2
        # Valore noto: Ix rispetto al diametro piatto = π·r⁴/8
        assert abs(Ix_flat - math.pi * 625 / 8) < 1e-6


# ─── Polygon ─────────────────────────────────────────────────────────────────

class TestPolygon:
    """Un quadrato 4×4 con vertice inferiore sinistro all'origine."""

    def _square(self):
        return Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], name="Quadrato")

    def test_area(self):
        p = self._square()
        assert abs(p.area() - 16) < EPS

    def test_centroid(self):
        p = self._square()
        xc, yc = p.centroid()
        assert abs(xc - 2.0) < EPS
        assert abs(yc - 2.0) < EPS

    def test_Ix_centroid(self):
        # b·h³/12 = 4·64/12 = 256/12
        p = self._square()
        rect = Rectangle(4, 4)
        assert abs(p.Ix_centroid() - rect.Ix_centroid()) < 1e-6

    def test_Iy_centroid(self):
        p = self._square()
        rect = Rectangle(4, 4)
        assert abs(p.Iy_centroid() - rect.Iy_centroid()) < 1e-6

    def test_Ixy_centroid_rectangle(self):
        # Per un rettangolo simmetrico Ixy_c = 0
        p = self._square()
        assert abs(p.Ixy_centroid()) < 1e-6

    def test_right_triangle(self):
        """Triangolo rettangolo come poligono vs classe Triangle."""
        b, h = 6.0, 4.0
        # Vertici CCW: angolo retto in origine
        poly = Polygon([(0, 0), (b, 0), (0, h)], name="Triangolo-poly")
        tri = Triangle(b, h)
        assert abs(poly.area() - tri.area()) < 1e-6
        px, py = poly.centroid()
        tx, ty = tri.centroid()
        assert abs(px - tx) < 1e-6
        assert abs(py - ty) < 1e-6
        assert abs(poly.Ix_centroid() - tri.Ix_centroid()) < 1e-6
        assert abs(poly.Iy_centroid() - tri.Iy_centroid()) < 1e-6

    def test_too_few_vertices(self):
        with pytest.raises(ValueError):
            Polygon([(0, 0), (1, 0)])


# ─── CrossSection ────────────────────────────────────────────────────────────

class TestCrossSection:
    def test_single_rectangle(self):
        """Una sezione con un solo rettangolo deve restituire i valori del rettangolo."""
        r = Rectangle(4, 6, 0, 0)
        sec = CrossSection()
        sec.add(r)
        p = sec.calculate()
        xc, yc = r.centroid()
        assert abs(p.area - r.area()) < EPS
        assert abs(p.xc - xc) < EPS
        assert abs(p.yc - yc) < EPS
        assert abs(p.Ix_c - r.Ix_centroid()) < EPS
        assert abs(p.Iy_c - r.Iy_centroid()) < EPS

    def test_hollow_rectangle(self):
        """Rettangolo pieno − rettangolo interno."""
        outer = Rectangle(10, 20, 0, 0)
        inner = Rectangle(6, 16, 2, 2)
        sec = CrossSection()
        sec.add(outer)
        sec.subtract(inner)
        p = sec.calculate()
        expected_A = outer.area() - inner.area()
        assert abs(p.area - expected_A) < EPS
        # Sezione simmetrica: baricentro al centro
        assert abs(p.xc - 5.0) < EPS
        assert abs(p.yc - 10.0) < EPS

    def test_T_section_barycentre(self):
        """Sezione a T: flangia 100×10 in cima, anima 10×90 sotto."""
        flange = Rectangle(100, 10, 0, 90)
        web = Rectangle(10, 90, 45, 0)
        sec = CrossSection()
        sec.add(flange)
        sec.add(web)
        p = sec.calculate()

        A_f = 100 * 10
        A_w = 10 * 90
        yc_f = 90 + 5    # = 95
        yc_w = 45
        A_tot = A_f + A_w
        Sy_expected = (A_f * 95 + A_w * 45) / A_tot

        assert abs(p.area - A_tot) < EPS
        assert abs(p.yc - Sy_expected) < 1e-6

    def test_empty_raises(self):
        sec = CrossSection()
        with pytest.raises(ValueError):
            sec.calculate()

    def test_remove(self):
        sec = CrossSection()
        sec.add(Rectangle(4, 4))
        sec.add(Circle(2))
        sec.remove(0)
        assert len(sec.shapes) == 1
        assert isinstance(sec.shapes[0], Circle)

    def test_save_load(self, tmp_path):
        fname = str(tmp_path / "test_sec.json")
        sec = CrossSection("Test")
        sec.add(Rectangle(4, 6, name="R1"))
        sec.subtract(Circle(1, xc=2, yc=3, name="C1"))
        sec.save(fname)

        loaded = CrossSection.load(fname)
        assert loaded.name == "Test"
        assert len(loaded.shapes) == 2
        assert loaded.shapes[0].name == "R1"
        assert loaded.shapes[1].sign == -1

        p1 = sec.calculate()
        p2 = loaded.calculate()
        assert abs(p1.area - p2.area) < EPS
        assert abs(p1.Ix_c - p2.Ix_c) < EPS

    def test_radius_of_gyration(self):
        """ix = sqrt(Ix_c / A)."""
        r = Rectangle(6, 10)
        sec = CrossSection()
        sec.add(r)
        p = sec.calculate()
        expected_ix = math.sqrt(r.Ix_centroid() / r.area())
        assert abs(p.ix - expected_ix) < EPS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
