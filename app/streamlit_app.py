"""Streamlit entry point for the Rostrum dashboard.

Placeholder: the dashboard is Phase 8. Run with ``make app``.

The screens to build, in order (see ``MASTER_SPEC_PLAN.md`` section 15.2):

1. **Upload** - audio (wav/mp3/m4a) plus transcript (file or paste); both are
   required by the brief. Also "choose a library speech" and "choose a sample
   flawed file" so a judge can see a result without recording anything.
2. **Processing** - step indicators: normalise, align, features, compare, explain.
3. **Analysis** - waveform with flaw regions coloured by rubric category; a
   shared-x Plotly overlay of participant vs envelope median with a shaded
   +/-1 sigma band for F0 (semitones), relative energy and local speech rate;
   the transcript with each word tinted by its combined deviation D_w;
   one explanation card per region (timestamp, label, severity, evidence table,
   rationale, fix, A/B playback); and a radar chart of the six categories with
   the confidence band.
4. **Export** - the analysis JSON and a PDF summary.

The mode badge is not optional: a norm-mode result must be visibly marked
lower-confidence, and a reject must say plainly that the audio and transcript
do not match.
"""

from __future__ import annotations


def main() -> None:
    """Render the dashboard."""
    import streamlit as st

    st.set_page_config(page_title="Rostrum", page_icon="🎙", layout="wide")
    st.title("Rostrum")
    st.caption("Contrastive speech analytics and temporal flaw grounding")
    st.info("The dashboard is not built yet (Phase 8). The analysis pipeline comes first.")


if __name__ == "__main__":
    main()
