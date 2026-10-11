"""The Rostrum dashboard (spec 15.2, prompt 22).

Run it with ``make app``, or::

    streamlit run app/streamlit_app.py --server.port 8501

Set ``ROSTRUM_API_URL`` to analyse through the FastAPI service instead of in
this process; nothing on screen changes, only where the numbers come from (see
:mod:`rostrum.dashboard.service`).

The seven screens, in the order a judge meets them:

1. **Sidebar** - upload audio *and* transcript, or pick a library text and a
   sample recording. Both halves are required: audio alone cannot be analysed,
   because every timestamp is anchored to a word of the transcript.
2. **Processing** - each pipeline stage and what it cost.
3. **Verdict** - the operating-mode badge, the score with its confidence band,
   the six-category radar, and the raw descriptors against the readers' range.
4. **Overlay** - the evidence: three panels sharing a time axis, this reading
   against the skilled readers' median and +/- 1 sigma band, flaw regions shaded
   by category.
5. **Transcript** - the same deviation mapped onto the words.
6. **Region cards** - one card per flaw: timestamp, severity, confidence,
   evidence table, rationale, fix, and A/B playback against a skilled reader.
7. **Export** - the analysis JSON and a PDF report.

Everything drawn here is read from the :class:`~rostrum.types.AnalysisResult`.
This file contains no measurement and no threshold: if a number appears on
screen, the pipeline computed it (CLAUDE.md invariant 12).
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("rostrum.app")

PAGE_TITLE = "Rostrum"
UPLOAD_TYPES = ["wav", "mp3", "m4a", "ogg", "flac", "opus"]


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #


def _state(st: Any) -> Any:
    """Initialise the keys the screens share, once per session."""
    defaults = {
        "result": None,  # AnalysisResult
        "stages": [],  # list[dict]
        "error": "",  # analysis failure, shown in place of the result
        "highlight": None,  # region_id picked out by "Jump"
        "transcript_text": "",  # the text area's contents
        "clip_cache": {},  # (region_id, which) -> wav bytes
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    return st.session_state


# --------------------------------------------------------------------------- #
# 1. Sidebar
# --------------------------------------------------------------------------- #


def _sidebar(st: Any, backend: Any, library: dict[str, Any]) -> tuple[Path | None, str, str | None]:
    """The upload / library picker.

    Returns:
        ``(audio_path, transcript, text_id)`` when the user pressed Analyse and
        both halves are present, else ``(None, "", None)``.
    """
    state = _state(st)
    st.sidebar.title("Rostrum")
    st.sidebar.caption("Contrastive speech analytics and temporal flaw grounding")

    source = st.sidebar.radio(
        "What do you want to analyse?",
        ["Upload a recording", "A sample from the dataset"],
        help=(
            "Uploading needs the transcript too. A dataset sample brings its own "
            "transcript, so you can see a full result without recording anything."
        ),
    )

    audio_path: Path | None = None
    transcript = ""
    text_id: str | None = None
    texts = {t["text_id"]: t for t in library.get("texts", [])}

    if source == "Upload a recording":
        upload = st.sidebar.file_uploader(
            "Audio (required)",
            type=UPLOAD_TYPES,
            help="wav / mp3 / m4a / ogg, up to 20 MB and 5 minutes.",
        )
        st.sidebar.markdown("**Transcript (required)**")
        transcript_file = st.sidebar.file_uploader(
            "...as a .txt file", type=["txt"], key="transcript_file"
        )
        if transcript_file is not None:
            state["transcript_text"] = transcript_file.getvalue().decode("utf-8", "replace")
        transcript = st.sidebar.text_area(
            "...or paste it",
            value=state["transcript_text"],
            height=160,
            help=(
                "Punctuation matters: every pause label and every context clause in "
                "the explanations is derived from it."
            ),
        )
        state["transcript_text"] = transcript

        if upload is not None:
            audio_path = _spool(upload)

        force = st.sidebar.selectbox(
            "Compare against a specific text?",
            ["Match automatically", *sorted(texts)],
            help=(
                "By default the transcript is matched against the library. Forcing a "
                "text is useful when your punctuation differs from ours."
            ),
        )
        text_id = None if force == "Match automatically" else force

    else:
        samples = library.get("samples", [])
        if not samples:
            st.sidebar.warning(
                "No sample recordings are on disk. Run `make data` to build the dataset."
            )
        else:
            labels = {
                f"{s['file_id']}  ({s.get('origin', '?')}"
                + (f", {s['label']} {s['level']}" if s.get("label") else "")
                + ")": s
                for s in samples
            }
            chosen = st.sidebar.selectbox("Sample recording", sorted(labels))
            sample = labels[chosen]
            audio_path = Path(sample["path"])
            text_id = sample.get("text_id")
            if text_id and text_id in texts:
                transcript = texts[text_id].get("transcript", "")
                state["transcript_text"] = transcript
                st.sidebar.caption(
                    f"Transcript taken from the library: {texts[text_id].get('title') or text_id}"
                )
            else:
                st.sidebar.warning(
                    "That sample has no library transcript; paste one below to analyse it."
                )
                transcript = st.sidebar.text_area("Transcript", height=140)

    st.sidebar.divider()
    ready = audio_path is not None and bool(transcript.strip())
    pressed = st.sidebar.button(
        "Analyse",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    )
    if not ready:
        missing = []
        if audio_path is None:
            missing.append("audio")
        if not transcript.strip():
            missing.append("a transcript")
        st.sidebar.caption("Still needed: " + " and ".join(missing) + ".")

    _sidebar_footer(st, backend, library)
    if pressed and ready:
        return audio_path, transcript, text_id
    return None, "", None


def _spool(upload: Any) -> Path:
    """Write an uploaded file to a temporary path the pipeline can open."""
    suffix = Path(upload.name).suffix or ".wav"
    # delete=False on purpose: the pipeline opens this path after the handle is
    # closed, so it must outlive the context manager. The OS temp directory is
    # the owner of the file from here on.
    directory = Path(tempfile.mkdtemp(prefix="rostrum_upload_"))
    # Keep the uploader's own stem: it becomes the result's file_id, and
    # "upload" would make every analysis look alike in an export.
    stem = Path(upload.name).stem or "recording"
    path = directory / f"{stem}{suffix}"
    path.write_bytes(upload.getvalue())
    return path


def _sidebar_footer(st: Any, backend: Any, library: dict[str, Any]) -> None:
    """What this deployment is, and what it can and cannot compare against."""
    with st.sidebar.expander("This deployment", expanded=False):
        st.caption(f"Analysis by: {getattr(backend, 'name', 'unknown')}")
        try:
            health = backend.health()
        except Exception as exc:
            st.error(f"Health check failed: {exc}")
            return
        st.write(
            {
                "status": health.get("status"),
                "code": health.get("code_version"),
                "config": str(health.get("config_hash", ""))[:12],
                "references": health.get("library_fingerprint"),
                "texts": health.get("texts"),
            }
        )
        for warning in list(library.get("warnings", []))[:6]:
            st.warning(warning, icon=":material/warning:")

    with st.sidebar.expander("Texts in the library", expanded=False):
        for text in library.get("texts", []):
            mode = text.get("mode", "norm")
            icon = ":material/check_circle:" if mode == "reference" else ":material/info:"
            st.markdown(
                f"{icon} **{text.get('title') or text['text_id']}** - "
                f"{text.get('n_references', 0)} reader(s), {mode} mode"
            )
        st.caption(
            f"A text needs {library.get('min_references', 0)} skilled readings before an "
            "upload of it is compared word by word; below that the comparison falls back "
            "to pooled norms."
        )


# --------------------------------------------------------------------------- #
# 2. Processing
# --------------------------------------------------------------------------- #


def _run_analysis(st: Any, backend: Any, audio: Path, transcript: str, text_id: str | None) -> None:
    """Run one analysis behind a stage-by-stage status panel."""
    from rostrum.dashboard.service import BackendError

    state = _state(st)
    state["error"] = ""
    state["highlight"] = None
    state["clip_cache"] = {}

    with st.status("Analysing...", expanded=True) as status:
        st.write("normalise -> check -> align -> features -> compare -> explain")
        try:
            result, stages = backend.analyze(audio, transcript, text_id)
        except BackendError as exc:
            status.update(label="The analysis could not be completed", state="error")
            state["result"] = None
            state["stages"] = []
            state["error"] = str(exc)
            return
        except Exception as exc:
            logger.exception("unexpected analysis failure")
            status.update(label="The analysis failed unexpectedly", state="error")
            state["result"] = None
            state["stages"] = []
            state["error"] = f"{type(exc).__name__}: {exc}"
            return

        state["result"] = result
        state["stages"] = stages
        total = sum(float(s.get("seconds", 0.0)) for s in stages)
        for stage in stages:
            st.write(
                f"**{stage['name']}** - {float(stage.get('seconds', 0.0)):.2f} s"
                + (f"  ({stage['note']})" if stage.get("note") else "")
            )
        status.update(
            label=(
                f"Done in {total:.1f} s - {result.mode} mode, " f"{len(result.regions)} region(s)"
            ),
            state="complete",
            expanded=False,
        )


def _stage_table(st: Any, stages: list[dict[str, Any]]) -> None:
    """The timings again, as a table, for anyone comparing runs."""
    if not stages:
        return
    with st.expander("Stage timings", expanded=False):
        st.dataframe(
            [
                {
                    "stage": s.get("name"),
                    "seconds": round(float(s.get("seconds", 0.0)), 3),
                    "note": s.get("note", ""),
                }
                for s in stages
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Timings are wall-clock and deliberately not part of the hashed result, so "
            "a slow machine cannot change an analysis."
        )


# --------------------------------------------------------------------------- #
# 3. Verdict
# --------------------------------------------------------------------------- #


def _verdict(st: Any, view: Any) -> None:
    """Mode badge, overall score with its band, radar, and descriptors."""
    from rostrum.dashboard.figures import radar_figure
    from rostrum.dashboard.palette import mode_badge

    result = view.result
    badge = mode_badge(result.mode, result.n_references, result.mode_reason)
    callout = {"success": st.success, "warning": st.warning, "error": st.error}[badge["tone"]]
    callout(f"**{badge['title']}** - {badge['detail']}")

    for warning in result.warnings[:8]:
        st.caption(f":material/info: {warning}")

    if result.mode == "reject":
        st.info(
            "Fix the transcript, or upload the recording that matches it, and analyse "
            "again. A score is withheld rather than guessed."
        )
        return

    left, right = st.columns([1.0, 1.25], gap="large")
    with left:
        overall = view.overall
        st.metric(
            "Overall",
            "n/a" if overall is None else f"{overall:.2f} / 10",
            delta=None if overall is None else f"+/- {view.overall_band:.2f}",
            delta_color="off",
            help=(
                "Mean of the six categories, each S_c = 10 exp(-k_c M_c). The band is "
                "the standard deviation over scores recomputed with each skilled "
                "reader left out, so it says how much the verdict depends on who "
                "happened to be in the envelope."
            ),
        )
        rows = view.category_scores()
        if rows:
            st.dataframe(
                [
                    {
                        "category": r["category"],
                        "score": round(r["score"], 2),
                        "+/-": round(r["band"], 2),
                        "regions": r["n_regions"],
                        "measurable": "no" if r["suppressed"] else "yes",
                    }
                    for r in rows
                ],
                hide_index=True,
                use_container_width=True,
            )
        for row in rows:
            if row["suppressed"]:
                st.caption(f":material/block: **{row['category']}**: {row['suppressed_reason']}")

    with right:
        st.plotly_chart(radar_figure(view), use_container_width=True, theme=None)

    _descriptors(st, view)


def _descriptors(st: Any, view: Any) -> None:
    """The raw descriptors with the skilled readers' range beside them."""
    rows = view.descriptor_rows()
    if not rows:
        return
    with st.expander("Raw descriptors against the skilled readers", expanded=False):
        st.dataframe(
            [
                {
                    "descriptor": r["descriptor"],
                    "this reading": _fmt(r["value"]),
                    "readers: low": _fmt(r["reference_low"]),
                    "median": _fmt(r["reference_median"]),
                    "high": _fmt(r["reference_high"]),
                    "n": r["n_references"],
                }
                for r in rows
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "The range is computed from the readers' own recordings, not from the "
            "envelope: the median of a per-word quantity is not the descriptor of any "
            "actual reading."
        )


def _fmt(value: Any) -> str:
    """One descriptor value, or a dash."""
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------- #
# 4. Overlay
# --------------------------------------------------------------------------- #


def _overlay(st: Any, view: Any) -> None:
    """The three-panel time-series overlay, and the waveform above it."""
    from rostrum.dashboard.figures import overlay_figure, waveform_figure

    state = _state(st)
    st.subheader("Where the delivery left the envelope")

    if view.result.mode == "norm":
        st.info(
            "Norm mode: there is a comparison but no per-word band to draw, because "
            "pooled norms are not curves of this text. The flagged regions below are "
            "still grounded in time.",
            icon=":material/info:",
        )

    samples = _participant_samples(view)
    if samples is not None:
        st.plotly_chart(
            waveform_figure(view, samples[0], samples[1]),
            use_container_width=True,
            theme=None,
        )

    st.plotly_chart(
        overlay_figure(view, highlight=state["highlight"]),
        use_container_width=True,
        theme=None,
    )
    with st.expander("How to read this", expanded=False):
        for panel in view.panels:
            st.markdown(f"**{panel.title}** ({panel.unit}) - {panel.why}")
        st.markdown(
            "The dashed line is the skilled readers' median and the shaded band is "
            "+/- 1 sigma, where sigma is 1.4826 x MAD with a floor, computed per word. "
            "The line breaks at every pause rather than bridging it. A region is "
            "entered when |z| > 2.5 and left when it falls below 1.5, then its edges "
            "are refined with a changepoint search (ruptures PELT, RBF cost) inside "
            "+/- 300 ms."
        )


def _participant_samples(view: Any) -> tuple[Any, int] | None:
    """The processed recording, for the waveform; ``None`` when it is not on disk."""
    from rostrum.clips import ClipStore
    from rostrum.io.audio import read_mono

    try:
        store = ClipStore.from_config(view.config)
        wav = store.processed_wav(view.result.analysis_id, file_id=view.result.file_id)
        samples, rate = read_mono(wav, sr=view.config.sample_rate)
    except Exception as exc:
        logger.info("no waveform available: %s", exc)
        return None
    return samples, rate


# --------------------------------------------------------------------------- #
# 5. Transcript
# --------------------------------------------------------------------------- #


def _transcript(st: Any, view: Any) -> None:
    """The transcript tinted by per-word deviation."""
    from rostrum.dashboard.transcript import deviation_legend_html, transcript_html

    st.subheader("The same deviation, word by word")
    if not view.has_deviation:
        st.info(
            "This analysis carries no per-word deviations, so the transcript is shown "
            "untinted. Re-run the analysis to produce them.",
            icon=":material/info:",
        )
    st.markdown(
        deviation_legend_html(view.deviation_cap, view.deviation_floor),
        unsafe_allow_html=True,
    )
    st.markdown(transcript_html(view), unsafe_allow_html=True)
    st.caption(
        "Tint is D = sqrt(sum_k lambda_k z_k^2) for that word - how far it sat from the "
        "skilled readers across every weighted feature at once. The ramp runs from an "
        f"ordinary word ({view.deviation_floor:.1f}) to every feature at the flag "
        f"threshold at once ({view.deviation_cap:.1f}). A coloured underline marks the "
        "words inside a flagged region; grey means nothing was measurable there, which "
        "is not the same as typical. Hover a word for its numbers."
    )


# --------------------------------------------------------------------------- #
# 6. Region cards
# --------------------------------------------------------------------------- #


def _cards(st: Any, view: Any, backend: Any) -> None:
    """One card per region, in time order, with A/B playback."""
    state = _state(st)
    st.subheader(f"Flaw regions ({len(view.regions)})")

    if not view.regions:
        st.success(
            "No region crossed the detection threshold. Every word stayed inside the "
            "skilled readers' envelope.",
            icon=":material/check_circle:",
        )
        return

    for region in view.regions:
        card = view.card_for(region.region_id)
        _card(st, view, backend, region, card, highlighted=state["highlight"] == region.region_id)


def _card(
    st: Any, view: Any, backend: Any, region: Any, card: dict[str, Any], *, highlighted: bool
) -> None:
    """Render one region card."""
    from rostrum.dashboard.palette import CATEGORY_LABELS, category_colour, level_label

    state = _state(st)
    category = CATEGORY_LABELS.get(region.category, region.category)
    title = (
        f"{region.t_start:.2f}-{region.t_end:.2f} s - {category}: {region.label} "
        f"- {level_label(region.level)}"
    )
    with st.container(border=True):
        head, jump = st.columns([5.0, 1.0], gap="small")
        with head:
            st.markdown(
                f"<span style='display:inline-block;width:10px;height:10px;"
                f"background:{category_colour(region.category)};border-radius:2px;"
                f"margin-right:8px;'></span><b>{title}</b>"
                + ("  &nbsp;:arrow_left: shown in the overlay" if highlighted else ""),
                unsafe_allow_html=True,
            )
            if card.get("headline"):
                st.markdown(f"**{card['headline']}**")
            if region.words:
                st.markdown(f"_&ldquo;{region.words}&rdquo;_")
        with jump:
            if st.button(
                "Jump",
                key=f"jump_{region.region_id}",
                help="Pick this region out in the overlay above.",
                use_container_width=True,
            ):
                state["highlight"] = region.region_id
                st.rerun()

        meta = st.columns(3)
        meta[0].metric("Confidence", f"{region.confidence:.2f}", help=_confidence_help(card))
        meta[1].metric("Severity", region.level, help=f"continuous severity s = {region.s:.3f}")
        meta[2].metric(
            "Readers",
            f"{region.envelope_n}",
            help="Skilled readings the comparison at this point was made against.",
        )

        if region.evidence:
            st.markdown("**The measurement**")
            st.dataframe(
                [
                    {
                        "feature": e.feature,
                        "this reading": f"{e.value:.3g} {e.unit}".strip(),
                        "readers' median": f"{e.env_median:.3g}",
                        "sigma": f"{e.env_sigma:.3g}",
                        "z": f"{e.z:+.2f}",
                        "n": e.n_references,
                    }
                    for e in region.evidence
                ],
                hide_index=True,
                use_container_width=True,
            )

        if region.context:
            st.markdown(f"**Context** - {region.context}")
        if region.rationale:
            st.markdown(f"**Why this is a flaw** - {region.rationale}")
        if region.fix:
            st.markdown(f"**What to do** - {region.fix}")
        if region.refined:
            st.caption(
                "Edges refined by changepoint search; before refinement this region was "
                f"{_fmt_time(region.t_start_unrefined)}-{_fmt_time(region.t_end_unrefined)} s."
            )
        if card.get("uncertain"):
            st.warning(
                "The rule and the classifier disagree here, so treat the label as "
                "uncertain; the timestamps and the measurements are unaffected.",
                icon=":material/help:",
            )

        _players(st, view, backend, region)


def _confidence_help(card: dict[str, Any]) -> str:
    """Where a card's confidence number came from."""
    basis = str(card.get("confidence_basis", "unavailable"))
    return {
        "leave_one_out": (
            "The fraction of skilled readers that still produce this flag when each one "
            "is left out of the envelope in turn."
        ),
        "classifier": "The classifier's calibrated probability for this label.",
        "unavailable": "No confidence basis was available for this region.",
    }.get(basis, basis)


def _fmt_time(value: Any) -> str:
    """A timestamp, or a dash."""
    return "-" if value is None else f"{float(value):.2f}"


def _players(st: Any, view: Any, backend: Any, region: Any) -> None:
    """A/B playback: this reading, against a skilled reader on the same words."""
    from rostrum.dashboard.service import BackendError

    state = _state(st)
    mine, theirs = st.columns(2, gap="medium")

    for column, which, heading in (
        (mine, "participant", "This reading"),
        (theirs, "reference", "A skilled reader, same words"),
    ):
        with column:
            st.markdown(f"**{heading}**")
            key = (int(region.region_id), which)
            if key not in state["clip_cache"]:
                try:
                    clip = backend.clip(view.result, region.t_start, region.t_end, which)
                except BackendError as exc:
                    state["clip_cache"][key] = ("error", str(exc))
                else:
                    audio = clip if isinstance(clip, bytes) else clip.to_wav_bytes()
                    note = "" if isinstance(clip, bytes) else clip.note
                    state["clip_cache"][key] = ("ok", (audio, note))

            kind, payload = state["clip_cache"][key]
            if kind == "error":
                st.caption(f":material/volume_off: {payload}")
                continue
            audio, note = payload
            st.audio(audio, format="audio/wav")
            if note:
                st.caption(note)


# --------------------------------------------------------------------------- #
# 7. Export
# --------------------------------------------------------------------------- #


def _export(st: Any, view: Any) -> None:
    """Download the analysis JSON and a PDF report."""
    from rostrum.dashboard.report import write_pdf

    st.subheader("Take it with you")
    result = view.result
    payload = json.dumps(result.to_dict(), indent=2).encode("utf-8")
    stem = f"{result.file_id}_{result.analysis_id[:8]}"

    left, right = st.columns(2, gap="medium")
    with left:
        st.download_button(
            "Download the analysis JSON",
            data=payload,
            file_name=f"{stem}.json",
            mime="application/json",
            use_container_width=True,
            help=(
                "The complete result: alignment, regions, evidence, scores and curves. "
                "This is the file the determinism check hashes."
            ),
        )
    with right:
        if st.button("Build the PDF report", use_container_width=True):
            with st.spinner("Rendering..."):
                out = Path(tempfile.gettempdir()) / f"{stem}.pdf"
                try:
                    write_pdf(view, out)
                except Exception as exc:
                    logger.exception("PDF export failed")
                    st.error(f"The PDF could not be rendered: {exc}")
                else:
                    st.session_state["pdf_bytes"] = out.read_bytes()
                    st.session_state["pdf_name"] = out.name
        if st.session_state.get("pdf_bytes"):
            st.download_button(
                "Download the PDF",
                data=st.session_state["pdf_bytes"],
                file_name=st.session_state.get("pdf_name", "rostrum_report.pdf"),
                mime="application/pdf",
                use_container_width=True,
            )

    st.caption(
        f"analysis {result.analysis_id[:16]}  |  config {result.config_hash[:12]}  |  "
        f"code {result.code_version or 'n/a'}  |  features v{result.features_version}  |  "
        f"{result.n_references} reference reading(s)"
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """Render the dashboard."""
    import streamlit as st

    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=":material/graphic_eq:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from rostrum.config import load_config
    from rostrum.dashboard.service import BackendError
    from rostrum.dashboard.view import AnalysisView

    state = _state(st)
    config = load_config()
    backend = _cached_backend(st, config)

    try:
        library = backend.library()
    except BackendError as exc:
        st.error(f"The analysis backend is not reachable: {exc}")
        st.stop()
        return

    audio, transcript, text_id = _sidebar(st, backend, library)
    if audio is not None:
        _run_analysis(st, backend, audio, transcript, text_id)

    st.title("Rostrum")
    st.caption(
        "Speech delivery measured against an envelope of skilled readings of the same "
        "text, with every flaw grounded in time and explained by the numbers behind it."
    )

    if state["error"]:
        st.error(state["error"])
    if state["result"] is None:
        _welcome(st, library)
        return

    view = AnalysisView(result=state["result"], config=config)
    _stage_table(st, state["stages"])
    _verdict(st, view)
    if view.result.mode != "reject":
        st.divider()
        _overlay(st, view)
        st.divider()
        _transcript(st, view)
        st.divider()
        _cards(st, view, backend)
    st.divider()
    _export(st, view)


def _cached_backend(st: Any, config: Any) -> Any:
    """One backend per session, so the reference library is loaded once."""
    from rostrum.dashboard.service import make_backend

    @st.cache_resource(show_spinner="Loading the reference library...")
    def build(config_hash: str) -> Any:
        return make_backend(config)

    return build(config.hash)


def _welcome(st: Any, library: dict[str, Any]) -> None:
    """What to do before anything has been analysed."""
    st.info(
        "Choose a recording in the sidebar - upload one with its transcript, or pick a "
        "sample from the dataset - then press **Analyse**.",
        icon=":material/arrow_back:",
    )
    reference = [t for t in library.get("texts", []) if t.get("mode") == "reference"]
    columns = st.columns(3, gap="large")
    columns[0].metric("Texts in the library", len(library.get("texts", [])))
    columns[1].metric(
        "With a full envelope",
        len(reference),
        help=(
            f"Texts with at least {library.get('min_references', 0)} skilled readings, "
            "which is what Reference mode needs."
        ),
    )
    columns[2].metric("Sample recordings", len(library.get("samples", [])))

    st.markdown("""
#### What you will get

| | |
|---|---|
| **Timestamped flaw regions** | Onset and offset in seconds, refined by changepoint search, for each of eight flaw types. |
| **The numbers behind each one** | The measured value, the skilled readers' median, the spread, and the resulting z-score. |
| **A six-category rubric score** | Pace, pausing, intonation, emphasis, energy, clarity - each with a confidence band. |
| **A/B playback** | The same words, as you read them and as a skilled reader read them. |

Timing comes from forced alignment (Montreal Forced Aligner), pitch from Praat,
and every score from the arithmetic above it. No language model takes part in
the judging.
""")


if __name__ == "__main__":
    main()
