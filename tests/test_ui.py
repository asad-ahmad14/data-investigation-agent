"""The web page is served by the API and keeps its safety property (no innerHTML)."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import STATIC_DIR, create_app

client = TestClient(create_app())


def test_root_redirects_to_the_page():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/ui/"


def test_page_and_assets_are_served():
    page = client.get("/ui/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "Data Investigation Agent" in page.text
    assert client.get("/ui/app.js").status_code == 200
    assert client.get("/ui/style.css").status_code == 200


def test_script_never_uses_innerhtml_or_eval():
    code = (Path(STATIC_DIR) / "app.js").read_text(encoding="utf-8")
    code_without_comments = "\n".join(
        line for line in code.splitlines() if not line.strip().startswith("//")
    )
    for forbidden in (".innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert forbidden not in code_without_comments


def test_page_hides_elements_with_the_hidden_attribute():
    # Regression: a flex/grid rule once overrode [hidden], leaving a stale "Investigating" bar.
    css = (Path(STATIC_DIR) / "style.css").read_text(encoding="utf-8")
    assert "[hidden] { display: none !important; }" in css


def test_api_docs_still_available():
    assert client.get("/docs").status_code == 200
