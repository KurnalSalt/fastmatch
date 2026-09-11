"""Command-line entry point: ``python -m fastmatch``.

Parses CLI arguments, configures the ``QApplication`` (OpenGL desktop attribute
must be set *before* the ``QApplication`` is created, per ``DESIGN.md`` §E.8),
optionally loads an image, builds the main window, and runs the event loop.

A non-GUI fast path exists:

* ``--generate-sample PATH`` writes a labelled synthetic image and exits without
  ever creating a ``QApplication`` (useful in CI / headless setups).

If no positional image is given, the app starts with an empty canvas; an image
is opened from *File ▸ Open Image…*.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""
    p = argparse.ArgumentParser(
        prog="fastmatch",
        description="GPU-accelerated visual pattern matcher.",
    )
    p.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Image file to open. If omitted, the app starts with an empty canvas.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "rocm", "cpu"],
        default="auto",
        help="Engine device preference (default: auto; canary-gated, CPU fallback).",
    )
    p.add_argument("--smoke-test", action="store_true", help="Check the packaged GUI and CPU matcher, then exit.")
    p.add_argument(
        "--generate-sample",
        metavar="PATH",
        default=None,
        help="Write a labelled synthetic sample to PATH and exit (no GUI).",
    )
    p.add_argument(
        "--w",
        type=int,
        default=12000,
        help="Sample width in px (default: 12000).",
    )
    p.add_argument(
        "--h",
        type=int,
        default=12000,
        help="Sample height in px (default: 12000).",
    )
    p.add_argument(
        "--n-targets",
        type=int,
        default=40,
        help="Number of motif stamps in a generated sample (default: 40).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for sample generation (default: 0).",
    )
    p.add_argument(
        "--max-ram-bytes",
        type=int,
        default=6_000_000_000,
        help="In-RAM RGB buffer cap before load falls back to a memmap "
        "(default: 6e9).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).
    """
    args = _build_parser().parse_args(argv)

    # Import the loader lazily so --help / arg errors don't pay import costs.
    from .loader import generate_sample, load_image

    # --- Headless sample-generation fast path (no GUI) ---------------------
    if args.generate_sample is not None:
        truth = generate_sample(
            args.generate_sample,
            w=args.w,
            h=args.h,
            n_targets=args.n_targets,
            seed=args.seed,
        )
        print(
            f"Wrote sample {args.generate_sample} "
            f"({args.w}x{args.h}, {len(truth)} motif stamps, seed={args.seed})."
        )
        return 0

    # --- GUI path ----------------------------------------------------------
    # Qt attributes that affect QApplication construction MUST be set before the
    # QApplication is instantiated (DESIGN.md §E.8).
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL, True)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])

    # Apply the persisted application theme before any window is built so the
    # first paint is already themed (no light->dark flash on startup).
    from .theme import apply_theme, load_theme

    apply_theme(app, load_theme(), persist=False)

    from .cpu import configure_threads, load_threads
    configure_threads(load_threads())

    from .i18n import load_language, set_language
    set_language(load_language(), persist=False)

    # Resolve the image: the user's file, or None to start on an empty canvas
    # (no auto-generated demo). An image is opened later via File > Open Image…
    doc = (
        load_image(args.image, max_ram_bytes=args.max_ram_bytes)
        if args.image is not None
        else None
    )

    # Build + show the window. Imported here so the headless path above never
    # imports the GUI stack.
    from .app import build_main_window

    win = build_main_window(doc, device=args.device)
    if args.smoke_test:
        import numpy as np
        from .engine import Matcher
        from .types import MatchParams
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QWidget
        rng = np.random.default_rng(9)
        pixels = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
        template = pixels[8:24, 8:24].copy()
        pixels[56:72, 56:72] = template
        matcher = Matcher(device="cpu")
        matcher.set_image(pixels)
        matches = matcher.match(template, MatchParams(), exclude_box=(8, 8, 16, 16))
        if not any(m.x == 56 and m.y == 56 and m.score > 0.99 for m in matches):
            raise RuntimeError("Packaged matcher smoke test failed")
        win._viewport.setViewport(QWidget())
        QTimer.singleShot(300, win.close)
        QTimer.singleShot(400, app.quit)
    win.resize(1280, 860)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
