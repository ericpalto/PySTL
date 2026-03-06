import pytest

from pystl import Predicate, export_formula


def test_formula_export_text_markdown_and_latex_tokens() -> None:
    p1 = Predicate("speed_ok")
    p2 = Predicate("alt_ok")
    phi = (p1 & ~p2).always((0, 2)).eventually((1, 3))

    text = phi.export("text")
    markdown = phi.export("markdown")
    latex = phi.export("latex")

    assert text == "eventually[1, 3](always[0, 2]((speed_ok and not (alt_ok))))"
    assert markdown == "F[1, 3](G[0, 2]((speed_ok and not (alt_ok))))"
    assert r"\square_{[0, 2]}" in latex
    assert r"\lozenge_{[1, 3]}" in latex
    assert r"speed\_ok" in latex
    assert r"alt\_ok" in latex


def test_formula_export_until_and_open_interval() -> None:
    p1 = Predicate("left")
    p2 = Predicate("right")
    phi = p1.until(p2, interval=(0, None))

    assert phi.export("text") == "(left) until[0, inf] (right)"
    assert phi.export("markdown") == "(left) U[0, inf] (right)"
    assert r"\mathcal{U}_{[0, \infty]}" in phi.export("latex")


def test_export_formula_aliases_and_validation() -> None:
    phi = Predicate("x").always((0, 1))

    assert export_formula(phi, format="md") == phi.export("markdown")
    assert export_formula(phi, format="plain") == phi.export("text")
    assert export_formula(phi, format="tex") == phi.export("latex")

    with pytest.raises(ValueError):
        phi.export("html")
