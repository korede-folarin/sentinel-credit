"""
Sentinel Credit, Dashboard Design System

One place for the palette, typography, chart template and the small set
of presentation components every page uses. Pages import from here and
never hard-code colours, so the product reads as one system.

Two registers, deliberately different:
- "terminal" (default): the dark charcoal intelligence views.
- "assessment": the Credit Risk Assessment page uses ivory decision
  surfaces so a reviewer's attention goes to the inputs and the result.

Status is never colour-only: every pill carries its text label.
"""

import html
import json
import os

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── PALETTE ───────────────────────────────────────────────────
CHARCOAL = "#151719"
GRAPHITE = "#202326"
GRAPHITE_LINE = "#2E3236"
IVORY = "#F2EBDD"
IVORY_TEXT = "#F5F1E9"
INK = "#1B1D20"           # text on ivory surfaces
STONE = "#A8A49D"
GOLD = "#B89A61"
OXBLOOD = "#741F2D"
EMERALD = "#438B73"
AMBER = "#C49A55"
CRIMSON = "#C4545C"

STATUS = {          # level -> (foreground, background)
    "positive": (EMERALD, "rgba(67,139,115,0.14)"),
    "warning": (AMBER, "rgba(196,154,85,0.14)"),
    "critical": (CRIMSON, "rgba(196,84,92,0.14)"),
    "neutral": (STONE, "rgba(168,164,157,0.12)"),
}

# Sequential series order for charts: one accent, then quiet neutrals.
SERIES = [GOLD, STONE, "#6F7A82", EMERALD, CRIMSON]

_FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Cormorant+Garamond:wght@500;600&"
    "family=Inter:wght@400;500;600&display=swap"
)

_CSS = f"""
<style>
@import url('{_FONTS}');

html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input, textarea, select {{
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}

/* Headings: editorial serif, used sparingly */
h1, .sc-display {{
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-weight: 600 !important; letter-spacing: 0.005em;
}}
h2, h3 {{ font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
          letter-spacing: -0.005em; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: {GRAPHITE}; border-right: 1px solid {GRAPHITE_LINE}; }}
[data-testid="stSidebarNav"] a span {{ font-size: 0.92rem; }}

/* Metrics: flat panels with a hairline rule, not rounded cards */
[data-testid="stMetric"] {{
    background: {GRAPHITE}; border: 1px solid {GRAPHITE_LINE};
    border-top: 2px solid {GOLD}; border-radius: 2px; padding: 0.9rem 1rem 0.8rem;
}}
[data-testid="stMetricLabel"] p {{
    color: {STONE} !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 0.08em;
}}
[data-testid="stMetricValue"] {{ font-size: 1.45rem !important; font-weight: 500; }}
/* Never hide a number behind an ellipsis: wrap instead */
[data-testid="stMetricValue"] div, [data-testid="stMetricLabel"] div,
[data-testid="stMetricLabel"] p {{
    white-space: normal !important; overflow: visible !important;
    text-overflow: clip !important; overflow-wrap: anywhere;
}}

/* Primary action: oxblood */
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button,
button[kind="secondaryFormSubmit"], button[kind="primaryFormSubmit"] {{
    background: {OXBLOOD} !important; border: 1px solid {OXBLOOD} !important;
    color: {IVORY_TEXT} !important; border-radius: 2px !important; font-weight: 600;
}}
.stButton button {{ border-radius: 2px !important; }}

hr {{ border-color: {GRAPHITE_LINE} !important; }}
[data-testid="stExpander"] {{ border: 1px solid {GRAPHITE_LINE}; border-radius: 2px; }}

/* Components */
.sc-kicker {{ color: {GOLD}; font-size: 0.72rem; letter-spacing: 0.16em;
              text-transform: uppercase; font-weight: 600; margin-bottom: 0.15rem; }}
.sc-title {{ font-family: 'Cormorant Garamond', Georgia, serif; font-size: 2.6rem;
             font-weight: 600; line-height: 1.05; margin: 0 0 0.35rem 0; color: {IVORY_TEXT}; }}
.sc-sub {{ color: {STONE}; font-size: 0.95rem; max-width: 780px; margin-bottom: 0.4rem; }}
.sc-rule {{ height: 1px; background: {GRAPHITE_LINE}; margin: 1.1rem 0 1.4rem; }}
.sc-section {{ font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase;
               color: {STONE}; font-weight: 600; margin: 1.6rem 0 0.6rem;
               padding-bottom: 0.35rem; border-bottom: 1px solid {GRAPHITE_LINE}; }}
.sc-pill {{ display: inline-block; padding: 0.18rem 0.6rem; border-radius: 2px;
            font-size: 0.74rem; font-weight: 600; letter-spacing: 0.06em;
            text-transform: uppercase; border: 1px solid; }}
.sc-note {{ color: {STONE}; font-size: 0.82rem; line-height: 1.5; }}
.sc-chip {{ display: inline-block; margin: 0 0.4rem 0.4rem 0; padding: 0.15rem 0.55rem;
            border: 1px solid {GRAPHITE_LINE}; border-radius: 2px; color: {STONE};
            font-size: 0.76rem; max-width: 100%; overflow-wrap: anywhere; }}
.sc-chip b {{ color: {IVORY_TEXT}; font-weight: 500; }}

/* Assessment register: ivory decision surfaces */
.sc-surface {{ background: {IVORY}; color: {INK}; padding: 1.4rem 1.6rem; border-radius: 2px;
               border-left: 3px solid {GOLD}; }}
.sc-surface .sc-kicker {{ color: #7A6338; }}
.sc-surface .sc-sub, .sc-surface .sc-note {{ color: #4A4740; }}
.sc-result-value {{ font-family: 'Cormorant Garamond', Georgia, serif; font-size: 3.2rem;
                    font-weight: 600; line-height: 1; color: {INK}; }}
.sc-result-label {{ font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
                    color: #4A4740; font-weight: 600; }}

@media (max-width: 760px) {{
    .sc-title {{ font-size: 2.0rem; }}
    .sc-result-value {{ font-size: 2.4rem; }}
    .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
}}
</style>
"""


def apply(page_title: str, layout: str = "wide") -> None:
    """First call on every page: page config, then the design system."""
    # "auto": open on desktop, collapsed on small screens so the sidebar
    # does not cover the page on a phone.
    st.set_page_config(page_title=f"{page_title} · Sentinel Credit", layout=layout,
                       initial_sidebar_state="auto")
    st.markdown(_CSS, unsafe_allow_html=True)
    _register_plotly_template()
    from utils import DEMO, DEMO_DATA_DIR
    if DEMO:
        snap = ""
        try:
            with open(os.path.join(DEMO_DATA_DIR, "MANIFEST.json"), encoding="utf-8") as f:
                snap = json.load(f).get("generated_at", "")[:10]
        except OSError:
            pass
        st.markdown(
            f"<div class='sc-note' style='border:1px solid {GOLD};padding:0.45rem 0.8rem;"
            "margin-bottom:0.8rem'><b>Public demo</b> · synthetic data · database views are a "
            f"snapshot{(' from ' + snap) if snap else ''}. Scoring of new applications is live. "
            "Existing-customer lookup and per-customer case studies run only in the local "
            "platform, because customer records are never published.</div>",
            unsafe_allow_html=True)


def header(kicker: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="sc-kicker">{html.escape(kicker)}</div>'
        f'<div class="sc-title">{html.escape(title)}</div>'
        + (f'<div class="sc-sub">{html.escape(subtitle)}</div>' if subtitle else "")
        + '<div class="sc-rule"></div>',
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="sc-section">{html.escape(title)}</div>', unsafe_allow_html=True)


def pill(text: str, level: str = "neutral") -> str:
    """Status label as HTML. Text is always shown; colour only reinforces it."""
    fg, bg = STATUS.get(level, STATUS["neutral"])
    return (f'<span class="sc-pill" style="color:{fg};background:{bg};border-color:{fg}">'
            f'{html.escape(text)}</span>')


def chips(items: dict) -> None:
    """Compact key/value context line (model, version, cutoff, ...)."""
    st.markdown("".join(
        f'<span class="sc-chip">{html.escape(str(k))} <b>{html.escape(str(v))}</b></span>'
        for k, v in items.items()), unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="sc-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def empty_state(what: str, how: str) -> None:
    """Honest empty state: says what is missing and what produces it."""
    st.markdown(
        f'<div class="sc-note" style="border:1px dashed {GRAPHITE_LINE};padding:0.9rem 1rem">'
        f'<b style="color:{IVORY_TEXT}">{html.escape(what)}</b><br>{html.escape(how)}</div>',
        unsafe_allow_html=True,
    )


# ── CHARTS ────────────────────────────────────────────────────
def _register_plotly_template() -> None:
    if "sentinel" in pio.templates:
        pio.templates.default = "sentinel"
        return
    axis = dict(gridcolor=GRAPHITE_LINE, zerolinecolor=GRAPHITE_LINE, linecolor=GRAPHITE_LINE,
                tickfont=dict(color=STONE, size=11), title=dict(font=dict(color=STONE, size=12)),
                showline=True, ticks="")
    pio.templates["sentinel"] = go.layout.Template(layout=dict(
        font=dict(family="Inter, system-ui, sans-serif", color=IVORY_TEXT, size=12),
        paper_bgcolor=GRAPHITE, plot_bgcolor=GRAPHITE,
        colorway=SERIES,
        title=dict(font=dict(size=14, color=IVORY_TEXT), x=0, xanchor="left", pad=dict(l=4)),
        xaxis=axis, yaxis=axis,
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(color=STONE, size=11),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=CHARCOAL, bordercolor=GOLD,
                        font=dict(family="Inter", color=IVORY_TEXT, size=12)),
        margin=dict(l=56, r=24, t=52, b=48),
    ))
    pio.templates.default = "sentinel"


def chart(fig: go.Figure, height: int = 340) -> None:
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
