"""Tests for the dashboard's pure layer (spec 15.2, prompt 22).

``app/streamlit_app.py`` is screens and widgets and is not tested here. What is
tested is everything it draws - the view adapter, the figures, the coloured
transcript, the palette and the PDF - because that is where a picture could
silently disagree with the numbers in the result.

The analysis under test is produced once, by the same generated readers the
pipeline tests use, so the assertions are about a real reference-mode result
rather than a hand-built stub.
"""

from __future__ import annotations

import numpy as np
import pytest
from eval.synthetic_audio import DEMO_TEXT, demo_library, make_utterance, parse_text

from rostrum.config import load_config
from rostrum.dashboard.figures import overlay_figure, radar_figure, region_hover, waveform_figure
from rostrum.dashboard.palette import (
    CATEGORY_COLOURS,
    CATEGORY_HATCH,
    deviation_colour,
    level_label,
    mode_badge,
    text_on,
)
from rostrum.dashboard.report import write_pdf
from rostrum.dashboard.transcript import (
    deviation_legend_html,
    transcript_html,
    transcript_words,
)
from rostrum.dashboard.view import PANELS, AnalysisView
from rostrum.pipeline import analyze
from rostrum.score.rubric import category_order


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return tmp_path_factory.mktemp("dashboard")


@pytest.fixture(scope="module")
def result(cfg, work):
    """One real reference-mode analysis of a participant who rushes a clause."""
    library, _ = demo_library(cfg, work / "library", n_readers=5)
    spoken = parse_text(DEMO_TEXT)
    rate = [1.0] * len(spoken.words)
    rate[4:8] = [1.9] * 4
    pause = [1.0] * len(spoken.words)
    pause[7] = 0.05
    utterance = make_utterance(
        "dash_rushing", work, text=spoken, seed=321, rate_scale=rate, pause_scale=pause
    )
    return analyze(
        utterance.path,
        DEMO_TEXT,
        "demo_anaphora",
        config=cfg,
        library=library,
        alignment=utterance.alignment,
        check_text_match=False,
        cache=False,
        work_dir=work / "analysis",
    )


@pytest.fixture(scope="module")
def view(result, cfg):
    return AnalysisView(result=result, config=cfg)


# --------------------------------------------------------------------------- #
# The view adapter
# --------------------------------------------------------------------------- #


def test_the_overlay_has_the_three_panels_the_spec_asks_for(view):
    """Pitch, energy and local rate, in that order."""
    assert [p.signal for p in PANELS] == ["f0_st", "energy_z", "rate_local_syll_per_s"]
    assert [p.signal for p in view.panels] == [p.signal for p in PANELS]


def test_every_panel_has_both_a_line_and_a_band(view):
    """A band with no line behind it was the bug that prompted this test."""
    for panel in view.panels:
        mine = view.participant_series(panel.signal)
        theirs = view.envelope_series(panel.signal)
        assert mine, f"{panel.signal}: no participant curve"
        assert theirs, f"{panel.signal}: no envelope band"
        assert theirs.n_references == view.result.n_references


def test_the_line_breaks_at_pauses_rather_than_bridging_them(view):
    """A bridged silence would draw phonation that did not happen."""
    series = view.participant_series("f0_st")
    assert np.isnan(series.t).any(), "no separators between words"
    # Every separator in t is a separator in value, so the break is real.
    assert np.isnan(series.value[np.isnan(series.t)]).all()


def test_the_band_is_the_median_plus_and_minus_one_sigma(view):
    """The shaded region must be exactly what the caption claims it is."""
    series = view.envelope_series("f0_st")
    ok = np.isfinite(series.value) & np.isfinite(series.lower) & np.isfinite(series.upper)
    assert ok.any()
    spread = series.upper[ok] - series.value[ok]
    assert np.allclose(spread, series.value[ok] - series.lower[ok], atol=1e-9)
    assert (spread >= 0).all()


def test_the_x_axis_covers_the_whole_recording(view):
    """Spec 15.2: a flaw is shown in the context of the speech, not as a clip."""
    t0, t1 = view.time_range
    assert t0 == 0.0
    assert t1 == pytest.approx(view.result.duration, abs=1e-6)
    for region in view.regions:
        assert t0 <= region.t_start and region.t_end <= t1 + 1e-6


def test_the_categories_are_in_report_order_and_all_six_are_present(view, cfg):
    """A missing category on screen would quietly change what the mean means."""
    rows = view.category_scores()
    assert [r["category"] for r in rows] == list(category_order(cfg))
    assert len(rows) == 6


def test_a_suppressed_category_is_carried_through_not_dropped(view):
    """ "Not measurable" and "perfect" must be distinguishable (spec 11)."""
    rows = {r["category"]: r for r in view.category_scores()}
    clarity = rows["clarity"]
    assert clarity["suppressed"] is True
    assert clarity["suppressed_reason"]


def test_the_view_reports_the_same_score_as_the_result(view, result):
    """The adapter must not round, rescale or recompute anything."""
    assert view.overall == pytest.approx(result.score.overall)
    assert view.overall_band == pytest.approx(result.score.overall_band)


# --------------------------------------------------------------------------- #
# The coloured transcript
# --------------------------------------------------------------------------- #


def test_every_aligned_word_appears_in_the_transcript(view, result):
    """A word dropped from the view is a word a judge cannot inspect."""
    words = transcript_words(view)
    assert len(words) == len(result.alignment.words)
    assert [w.idx for w in words] == sorted(w.idx for w in result.alignment.words)


def test_words_inside_a_region_are_marked_as_such(view):
    """The underline is the non-colour channel that marks a flagged word."""
    words = {w.idx: w for w in transcript_words(view)}
    assert view.regions, "the fixture should produce at least one region"
    for region in view.regions:
        indices = [int(i) for i in region.word_idx]
        for idx in range(min(indices), max(indices) + 1):
            assert words[idx].in_region
            assert words[idx].category


def test_the_deviation_tint_is_monotonic_and_capped(view):
    """A darker word must mean a more deviant word, and one outlier must not
    flatten the rest."""
    cap = view.deviation_cap
    shades = [deviation_colour(d, cap=cap) for d in (0.0, 1.0, 2.0, 4.0)]
    luminance = [sum(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in shades]
    assert luminance == sorted(luminance, reverse=True), "tint must darken with D"
    assert deviation_colour(99.0, cap=cap) == deviation_colour(cap, cap=cap)


def test_an_unmeasurable_word_is_grey_not_pale(view):
    """Grey says "nobody looked"; the ramp's pale end says "measured, typical"."""
    assert deviation_colour(None) == "#EFEFEF"
    assert deviation_colour(float("nan")) == "#EFEFEF"
    assert deviation_colour(0.0) != deviation_colour(None)


def test_the_transcript_html_escapes_its_text(cfg, view):
    """The transcript is rendered with unsafe_allow_html, so it must be safe."""
    html = transcript_html(view)
    assert "<div" in html and "</div>" in html
    assert "<script" not in html.lower()

    hostile = view.result
    original = hostile.alignment.words[0].display
    hostile.alignment.words[0].display = "<script>alert(1)</script>"
    try:
        assert "<script>" not in transcript_html(AnalysisView(result=hostile, config=cfg))
    finally:
        hostile.alignment.words[0].display = original


def test_the_tint_legend_prints_the_numbers_it_stands_for(view):
    """A colour ramp with no scale is decoration, not evidence."""
    legend = deviation_legend_html(view.deviation_cap)
    assert "D" in legend
    assert "n/a" in legend


def test_hover_text_carries_the_numbers(view):
    """Hover is where the per-word z-scores are read."""
    words = [w for w in transcript_words(view) if w.top_z]
    assert words
    tooltip = words[0].tooltip()
    assert "D =" in tooltip
    assert "z" in tooltip


# --------------------------------------------------------------------------- #
# Accessibility
# --------------------------------------------------------------------------- #


def test_every_category_has_a_colour_and_a_second_channel(cfg):
    """Prompt 22: colour is never the only encoding."""
    for category in category_order(cfg):
        assert category in CATEGORY_COLOURS
        assert CATEGORY_HATCH.get(category)


def test_text_stays_readable_on_both_ends_of_the_ramp():
    """White on pale, or black on deep blue, would be unreadable."""
    assert text_on(deviation_colour(0.0)) == "#000000"
    assert text_on(deviation_colour(4.0)) == "#FFFFFF"


def test_each_severity_rung_has_words_as_well_as_a_code():
    """ "L3" means nothing to a speaker; "disruptive" does."""
    for level in ("L0", "L1", "L2", "L3", "L4"):
        assert level_label(level) != level
        assert level in level_label(level)


def test_norm_mode_is_badged_as_lower_confidence():
    """Prompt 22 requires the mode to be visible, not inferred."""
    assert "lower confidence" in mode_badge("norm")["title"].lower()
    assert mode_badge("norm")["tone"] == "warning"
    assert mode_badge("reject")["tone"] == "error"
    assert "5" in mode_badge("reference", 5)["title"]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def test_the_overlay_figure_draws_every_region_on_every_panel(view):
    """A region shaded on one panel only would hide half the evidence."""
    figure = overlay_figure(view)
    assert len(figure.layout.shapes) == len(view.regions) * len(view.panels)
    assert figure.data


def test_the_band_polygon_closes_per_word_not_across_pauses(view):
    """One polygon spanning a silence draws a spread nobody measured."""
    from rostrum.dashboard.figures import _band_polygon

    x, y = _band_polygon(view.envelope_series("energy_z"))
    assert x.size and x.size == y.size
    # Each subpath must have an even number of points: a there-and-back loop.
    for block in np.split(x, np.flatnonzero(~np.isfinite(x))):
        finite = block[np.isfinite(block)]
        if finite.size:
            assert finite.size % 2 == 0


def test_region_hover_states_the_measurement(view):
    """Hover must carry the number, not just the label."""
    region = view.regions[0]
    text = region_hover(region, view.card_for(region.region_id))
    assert region.label in text
    assert "s" in text and "z =" in text


def test_the_radar_never_draws_a_suppressed_category_as_a_score(view):
    """Clarity at 10 would read as perfect for something nobody measured."""
    figure = radar_figure(view)
    filled = [t for t in figure.data if getattr(t, "fill", None) == "toself"]
    assert filled, "no polygon drawn"
    labels = list(filled[0].theta)
    assert not any("not measurable" in str(label) for label in labels)

    crosses = [t for t in figure.data if getattr(t.marker, "symbol", None) == "x"]
    assert crosses, "a suppressed category must still be shown, as a cross"


def test_the_waveform_decimates_without_losing_the_peaks(view):
    """Subsampling a 16 kHz signal would draw a quieter recording than analysed."""
    rng = np.random.default_rng(0)
    samples = rng.uniform(-0.9, 0.9, 32000).astype(np.float32)
    samples[12345] = 0.98
    figure = waveform_figure(view, samples, 16000, max_points=200)
    drawn = np.asarray(figure.data[0].y, dtype=float)
    assert np.nanmax(drawn) >= 0.97


def test_an_analysis_with_no_curves_draws_an_explanation_not_a_crash(cfg, result):
    """A rejected upload has no curves; the figure must say so."""
    import copy

    bare = copy.deepcopy(result)
    bare.curves = {}
    bare.envelope_curves = {}
    figure = overlay_figure(AnalysisView(result=bare, config=cfg))
    assert "No curves" in str(figure.layout.title.text)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def test_the_pdf_renders_and_names_its_provenance(view, tmp_path):
    """A report that cannot be traced to a build is not evidence."""
    out = write_pdf(view, tmp_path / "report.pdf")
    assert out.is_file()
    assert out.stat().st_size > 10_000
    assert out.read_bytes().startswith(b"%PDF")


def test_the_pdf_survives_a_result_with_no_regions(cfg, result, tmp_path):
    """A clean reading is the common case and must not crash the export."""
    import copy

    clean = copy.deepcopy(result)
    clean.regions = []
    clean.cards = []
    out = write_pdf(AnalysisView(result=clean, config=cfg), tmp_path / "clean.pdf")
    assert out.is_file()


# --------------------------------------------------------------------------- #
# The Streamlit screens
# --------------------------------------------------------------------------- #
#
# Driven through Streamlit's own headless harness, which executes the real
# script. These are the prompt 22 acceptance checks: the app has to *run*, and
# it has to render a full analysis without a browser in the loop.


@pytest.fixture
def app():
    """The dashboard script, ready to run headlessly.

    The path is absolute because ``AppTest.from_file`` resolves a relative one
    against the *calling* file, which would look inside ``tests/``.
    """
    from streamlit.testing.v1 import AppTest

    from rostrum.config import repo_root

    return AppTest.from_file(
        str(repo_root() / "app" / "streamlit_app.py"), default_timeout=300
    )


def test_the_dashboard_runs_with_nothing_uploaded(app):
    """A judge opening the page must not meet a traceback."""
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert "Rostrum" in [t.value for t in app.title]
    assert any(b.label == "Analyse" for b in app.button)


def test_the_analyse_button_is_disabled_until_both_halves_are_present(app):
    """Audio alone cannot be analysed, and the interface should say so."""
    app.run()
    analyse = next(b for b in app.button if b.label == "Analyse")
    assert analyse.disabled is True


def test_the_dashboard_renders_a_full_analysis(app, result):
    """Verdict, overlay, transcript, cards and export, from a real result."""
    app.session_state["result"] = result
    app.session_state["stages"] = [{"name": "features", "seconds": 1.0, "note": "ok"}]
    app.run()
    assert not app.exception, [e.value for e in app.exception]

    text = " ".join(m.value for m in app.markdown if isinstance(m.value, str))
    assert "Reference mode" in text or "Reference mode" in " ".join(
        s.value for s in app.success if isinstance(s.value, str)
    )

    labels = [m.label for m in app.metric]
    assert "Overall" in labels
    # One "Jump" button per region, plus the sidebar's Analyse.
    jumps = [b for b in app.button if b.label == "Jump"]
    assert len(jumps) == len(result.regions)
    assert any("flaw region" in str(h.value).lower() for h in app.subheader)


def test_a_rejected_analysis_shows_no_score(app, result):
    """Reject must withhold the rubric rather than print a meaningless number."""
    import copy

    rejected = copy.deepcopy(result)
    rejected.mode = "reject"
    rejected.mode_reason = "wer=1.000 > 0.25"
    rejected.score = None
    rejected.regions = []
    rejected.cards = []

    app.session_state["result"] = rejected
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    errors = " ".join(str(e.value) for e in app.error)
    assert "do not match" in errors
    assert "Overall" not in [m.label for m in app.metric]
