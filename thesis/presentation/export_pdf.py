"""Check the rendered slide structure and export it with installed Google Chrome.

Run after Quarto: uv run --with playwright==1.63.0 python
thesis/presentation/export_pdf.py
"""

from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    deck = Path(__file__).with_name("defence.html")
    destination = deck.with_suffix(".pdf")
    with sync_playwright() as playwright:
        # Chrome is available on the author's Mac and GitHub's Ubuntu runner.
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        # The audience deck must work without a network connection.
        page.route("http://**/*", lambda route: route.abort())
        page.route("https://**/*", lambda route: route.abort())
        page.goto(deck.as_uri())
        page.wait_for_function("window.Reveal && Reveal.isReady()")
        counts = page.evaluate("""() => ({
            slides: Reveal.getSlides().length,
            timing: Reveal.getSlides().filter(s => s.hasAttribute('data-timing'))
                .map(s => Number(s.dataset.timing)),
            notes: document.querySelectorAll('.slides aside.notes').length
        })""")
        if (
            counts["slides"] != 10
            or counts["notes"] != 10
            or len(counts["timing"]) != 7
            or sum(counts["timing"]) != 300
        ):
            raise ValueError(
                f"Expected seven timed slides plus three backups: {counts}"
            )

        page.goto(deck.as_uri() + "?print-pdf")
        page.wait_for_function("window.Reveal && Reveal.isReady()")
        page.wait_for_selector(".pdf-page")
        page.evaluate("document.fonts.ready")
        page.wait_for_function("""() => [...document.images].every(
            image => image.complete && image.naturalWidth > 0)""")
        if page.locator(".pdf-page").count() != 10 or errors:
            raise ValueError(f"PDF layout or JavaScript error: {errors}")
        page.pdf(
            path=str(destination), print_background=True, prefer_css_page_size=True
        )
        browser.close()
    print(  # noqa: T201 -- command-line export confirmation
        f"Checked 10 slides, 10 speaker notes, 300 seconds. Exported {destination}"
    )


if __name__ == "__main__":
    main()
