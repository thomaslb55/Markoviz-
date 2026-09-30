"""
Génération d'un rapport PDF automatique (synthèse d'allocation).

Utilise ReportLab (mise en page) + Kaleido (export statique des figures
Plotly en PNG, intégrées au PDF).
"""

from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

ALLIANZ_BLUE = colors.HexColor("#003781")
ALLIANZ_TURQUOISE = colors.HexColor("#0091C7")


def _fig_to_image(fig: go.Figure, width_px: int = 1000, height_px: int = 600) -> Image:
    """Exporte une figure Plotly en PNG (via kaleido) et retourne un objet
    Image ReportLab prêt à insérer dans le document."""
    png_bytes = fig.to_image(format="png", width=width_px, height=height_px, scale=2)
    buf = io.BytesIO(png_bytes)
    return Image(buf, width=16 * cm, height=16 * cm * height_px / width_px)


RATIO_METRIC_LABELS = {
    "Ratio de Sharpe", "Ratio de diversification", "Expected TVPI", "Expected DPI",
}


def format_metric_cell(row_label: str, value) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    if any(label in str(row_label) for label in RATIO_METRIC_LABELS):
        return f"{value:.2f}"
    if abs(value) <= 1.5:
        return f"{value:.2%}"
    return f"{value:.2f}"


def _dataframe_to_table(df: pd.DataFrame, col_widths: list | None = None) -> Table:
    data = [[""] + list(df.columns)] if df.index.name is None else [[df.index.name] + list(df.columns)]
    for idx, row in df.iterrows():
        formatted = [_format_cell(idx, v) for v in row]
        data.append([str(idx)] + formatted)

    table = Table(data, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ALLIANZ_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ]
        )
    )
    return table


def generate_report(
    output_path: str,
    title: str,
    weights: pd.Series,
    risk_summary: pd.DataFrame,
    risk_contribution: pd.DataFrame,
    efficient_frontier_fig: go.Figure | None = None,
    allocation_fig: go.Figure | None = None,
    correlation_fig: go.Figure | None = None,
    methodology_notes: list[str] | None = None,
) -> None:
    """Assemble un rapport PDF de synthèse et l'écrit à `output_path`."""
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleAllianz", parent=styles["Title"], textColor=ALLIANZ_BLUE)
    heading_style = ParagraphStyle("HeadingAllianz", parent=styles["Heading2"], textColor=ALLIANZ_BLUE)
    body_style = styles["BodyText"]

    doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    elements = []

    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 0.5 * cm))
    elements.append(
        Paragraph(
            "Rapport généré automatiquement — modèle de frontière efficiente pour portefeuille "
            "d'actifs privés. Document à usage interne, ne constitue pas un conseil en investissement.",
            body_style,
        )
    )
    elements.append(Spacer(1, 1 * cm))

    elements.append(Paragraph("Allocation optimale", heading_style))
    elements.append(_dataframe_to_table(weights.to_frame("Poids")))
    elements.append(Spacer(1, 0.5 * cm))
    if allocation_fig is not None:
        elements.append(_fig_to_image(allocation_fig))
    elements.append(PageBreak())

    elements.append(Paragraph("Mesures de risque", heading_style))
    elements.append(_dataframe_to_table(risk_summary))
    elements.append(Spacer(1, 0.5 * cm))
    elements.append(Paragraph("Contribution au risque par classe d'actifs", heading_style))
    elements.append(_dataframe_to_table(risk_contribution))
    elements.append(PageBreak())

    if efficient_frontier_fig is not None:
        elements.append(Paragraph("Frontière efficiente", heading_style))
        elements.append(_fig_to_image(efficient_frontier_fig))
        elements.append(PageBreak())

    if correlation_fig is not None:
        elements.append(Paragraph("Matrice de corrélation", heading_style))
        elements.append(_fig_to_image(correlation_fig))
        elements.append(PageBreak())

    if methodology_notes:
        elements.append(Paragraph("Méthodologie et limites", heading_style))
        for note in methodology_notes:
            elements.append(Paragraph(f"• {note}", body_style))
            elements.append(Spacer(1, 0.2 * cm))

    doc.build(elements)
