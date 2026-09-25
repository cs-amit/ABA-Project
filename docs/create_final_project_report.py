from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "Smart-Sleep-Alarm-Project-Report-IIM-Jammu.docx"
BUILD = ROOT / ".report-build"
PHONE_IMAGE = ROOT / "phone-screen.png"
WATCH_IMAGE = ROOT / "watch-screen.png"

NAVY = "14213D"
BLUE = "1F5D8F"
TEAL = "2A9D8F"
GOLD = "E9C46A"
PALE = "EAF1F7"
LIGHT = "F5F7FA"
MID = "6B7280"
GRID = "D9D9D9"
BLACK = "000000"


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, color=GRID, size="6"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_text(cell, text, bold=False, color=BLACK, size=9.2, align=WD_ALIGN_PARAGRAPH.LEFT):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    r = p.add_run(str(text))
    r.bold = bold
    r.font.name = "Aptos"
    r.font.size = Pt(size)
    r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_margins(cell)
    set_cell_border(cell)


def add_table(doc, headers, rows, widths=None, font_size=9.0):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    for i, header in enumerate(headers):
        set_cell_text(hdr.cells[i], header, bold=True, color="FFFFFF", size=9.1,
                      align=WD_ALIGN_PARAGRAPH.CENTER)
        shade(hdr.cells[i], NAVY)
        if widths:
            hdr.cells[i].width = Inches(widths[i])
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            align = WD_ALIGN_PARAGRAPH.CENTER if i > 0 and len(str(value)) < 24 else WD_ALIGN_PARAGRAPH.LEFT
            set_cell_text(cells[i], value, size=font_size, align=align)
            if ridx % 2:
                shade(cells[i], PALE)
            if widths:
                cells[i].width = Inches(widths[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def set_run_font(run, name="Aptos", size=11, bold=False, color=BLACK, italic=False):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def add_body(doc, text, bold_lead=None, size=10.7, after=6, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.12
    if bold_lead and text.startswith(bold_lead):
        a, b = text[:len(bold_lead)], text[len(bold_lead):]
        set_run_font(p.add_run(a), bold=True, size=size)
        set_run_font(p.add_run(b), size=size)
    else:
        set_run_font(p.add_run(text), size=size)
    return p


def add_bullets(doc, items, size=10.4, after=3):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.left_indent = Inches(0.22)
        p.paragraph_format.first_line_indent = Inches(-0.14)
        p.paragraph_format.space_after = Pt(after)
        p.paragraph_format.line_spacing = 1.06
        set_run_font(p.add_run(item), size=size)


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    set_run_font(p.add_run(text), size=8.6, italic=True, color=MID)


def add_picture(doc, path, width, caption=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Inches(width))
    if caption:
        add_caption(doc, caption)


def add_chapter_title(doc, chapter, title, kicker=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run(chapter.upper())
    set_run_font(r, size=9, bold=True, color=BLUE)
    p = doc.add_paragraph(style="Heading 1")
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(8)
    set_run_font(p.add_run(title), name="Aptos Display", size=23, bold=True)
    if kicker:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(11)
        set_run_font(p.add_run(kicker), size=11.2, italic=True, color=MID)


def add_subheading(doc, text):
    p = doc.add_paragraph(style="Heading 2")
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    set_run_font(p.add_run(text), name="Aptos Display", size=14, bold=True)


def page_break(doc):
    doc.add_page_break()


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)
    set_run_font(run, size=8.5, color=MID)


def create_visuals():
    BUILD.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.patch.set_facecolor("#14213D")
    ax.set_facecolor("#14213D")
    x = np.linspace(0, 10, 700)
    envelope = 0.85 - 0.045 * x
    wave = envelope * np.sin(2.35 * x) + 0.11 * np.sin(8.5 * x)
    ax.plot(x, wave, color="#8FD9D1", lw=2.5)
    ax.fill_between(x, wave, -1.2, color="#1F5D8F", alpha=.25)
    ax.axvspan(7.3, 9.4, color="#E9C46A", alpha=.18)
    ax.scatter([8.65], [wave[np.abs(x-8.65).argmin()]], s=130, color="#E9C46A", edgecolor="white", lw=1.5, zorder=5)
    ax.text(8.35, .95, "WAKE WINDOW", color="#E9C46A", weight="bold", fontsize=12, ha="center")
    ax.text(.2, -1.02, "Wearable signal", color="white", fontsize=11, weight="bold")
    ax.text(9.8, -1.02, "Safe alarm decision", color="white", fontsize=11, weight="bold", ha="right")
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.22, 1.22)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(BUILD / "cover_visual.png", dpi=220, bbox_inches="tight", facecolor="#14213D")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.axis("off")
    labels = ["Galaxy Watch4\nSensors", "Wear OS\nCapture", "Data Layer\nTransfer",
              "Android\nStorage", "Feature and\nModel Pipeline", "Wake Decision\nand Alarm"]
    xs = np.linspace(0.07, 0.93, len(labels))
    colors = ["#2A9D8F", "#3A86A8", "#4776A8", "#5A67A5", "#7B61A8", "#E76F51"]
    for i, (x, label, color) in enumerate(zip(xs, labels, colors)):
        ax.add_patch(plt.Rectangle((x - 0.07, 0.36), 0.14, 0.29, transform=ax.transAxes,
                                   facecolor=color, edgecolor="none"))
        ax.text(x, 0.505, label, ha="center", va="center", color="white", weight="bold",
                transform=ax.transAxes, fontsize=9)
        if i < len(labels) - 1:
            ax.annotate("", xy=(xs[i+1]-0.075, 0.505), xytext=(x+0.075, 0.505),
                        xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color="#4B5563", lw=1.8))
    ax.text(0.5, 0.18, "Exact-time phone fallback remains independent of watch connectivity and model output",
            ha="center", va="center", color="#14213D", weight="bold", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(BUILD / "architecture.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), gridspec_kw={"width_ratios": [1, 1.45]})
    split_counts = [29, 9, 9]
    axes[0].pie(split_counts, labels=["Train 29", "Validation 9", "Test 9"], startangle=90,
                colors=["#1F5D8F", "#2A9D8F", "#E9C46A"], wedgeprops=dict(width=0.42, edgecolor="white"),
                textprops={"fontsize": 9})
    axes[0].set_title("Subject-disjoint split", weight="bold", color="#14213D")
    stage_counts = [16381, 19359]
    axes[1].barh(["Light sleep", "Other stages"], stage_counts, color=["#2A9D8F", "#A9B8C6"])
    axes[1].set_title("Held-out test epochs", weight="bold", color="#14213D")
    axes[1].grid(axis="x", alpha=.18)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    for i, value in enumerate(stage_counts):
        axes[1].text(value + 300, i, f"{value:,}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(BUILD / "dataset_profile.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    names = ["Baseline\nlogistic", "CNN GRU", "CNN LSTM", "Raw CNN GRU", "Improved\nlogistic"]
    f1 = [0.5521, 0.4750, 0.5152, 0.6234, 0.6260]
    auc = [0.5731, 0.5266, 0.5248, 0.5146, 0.5724]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    width = .34
    ax.bar(x - width/2, f1, width, label="F1", color="#1F5D8F")
    ax.bar(x + width/2, auc, width, label="ROC AUC", color="#2A9D8F")
    ax.set_ylim(0.4, 0.68)
    ax.set_xticks(x, names)
    ax.set_ylabel("Held-out score")
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    for xpos, value in zip(x - width/2, f1):
        ax.text(xpos, value + .008, f"{value:.3f}", ha="center", fontsize=8)
    for xpos, value in zip(x + width/2, auc):
        ax.text(xpos, value + .008, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(BUILD / "model_comparison.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    cms = {
        "Baseline logistic": [[9287, 10072], [6293, 10088]],
        "CNN GRU": [[10430, 8929], [8498, 7883]],
        "CNN LSTM": [[8304, 11055], [6860, 9521]],
    }
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (name, cm) in zip(axes, cms.items()):
        arr = np.array(cm)
        ax.imshow(arr, cmap="Blues")
        ax.set_title(name, fontsize=10, weight="bold")
        ax.set_xticks([0, 1], ["Other", "Light"], fontsize=8)
        ax.set_yticks([0, 1], ["Other", "Light"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Actual", fontsize=8)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{arr[i,j]:,}", ha="center", va="center",
                        color="white" if arr[i,j] > arr.max()*.72 else "#14213D", fontsize=9, weight="bold")
    fig.tight_layout()
    fig.savefig(BUILD / "confusion_matrices.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    ax.axis("off")
    steps = [("1", "Capture", "Motion and\nheart signals"), ("2", "Clean", "Validity and\noff-body checks"),
             ("3", "Epoch", "Causal 30-second\nwindows"), ("4", "Sequence", "Ten prior\nepochs"),
             ("5", "Estimate", "Light-sleep\nprobability"), ("6", "Decide", "Trigger\nor wait")]
    for idx, (num, title, sub) in enumerate(steps):
        x = .06 + idx*.17
        ax.add_patch(plt.Circle((x, .62), .045, transform=ax.transAxes, color="#1F5D8F"))
        ax.text(x, .62, num, transform=ax.transAxes, color="white", weight="bold", ha="center", va="center")
        ax.text(x, .42, title, transform=ax.transAxes, color="#14213D", weight="bold", ha="center")
        ax.text(x, .25, sub, transform=ax.transAxes, color="#4B5563", ha="center", va="center",
                fontsize=7.3, linespacing=1.25)
        if idx < 5:
            ax.annotate("", xy=(x+.12, .62), xytext=(x+.055, .62), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", color="#9CA3AF", lw=1.8))
    fig.tight_layout()
    fig.savefig(BUILD / "method_flow.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_document():
    OUT.parent.mkdir(exist_ok=True)
    create_visuals()
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(.62)
    sec.bottom_margin = Inches(.58)
    sec.left_margin = Inches(.72)
    sec.right_margin = Inches(.72)
    sec.header_distance = Inches(.22)
    sec.footer_distance = Inches(.28)

    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    normal.font.size = Pt(10.7)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12
    for name in ("Title", "Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[name]
        style.font.name = "Aptos Display"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Aptos Display")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos Display")
        style.font.color.rgb = RGBColor(0, 0, 0)

    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_run_font(header.add_run("SMART SLEEP ALARM  |  PROJECT REPORT"), size=7.8, bold=True, color=MID)
    footer = sec.footer.paragraphs[0]
    set_run_font(footer.add_run("Indian Institute of Management Jammu"), size=8.2, color=MID)
    footer.add_run(" " * 58)
    add_page_number(footer)

    # Page 1 Cover
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(24)
    p.paragraph_format.space_after = Pt(4)
    set_run_font(p.add_run("SMART SLEEP ALARM"), name="Aptos Display", size=30, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    set_run_font(p.add_run("Wearable sensing and machine learning for a safer wake window"), size=13, color=BLUE)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(15)
    set_run_font(p.add_run("PROJECT REPORT"), size=10, bold=True, color=MID)
    add_picture(doc, BUILD / "cover_visual.png", 6.75)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(12)
    set_run_font(p.add_run("Wearable sensing, on-device inference and an exact-time fallback"), size=10.8, color=MID)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    set_run_font(p.add_run("Indian Institute of Management Jammu"), size=15, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    set_run_font(p.add_run("MBA Project Submission  |  September 2026"), size=10.5, color=MID)

    # Page 2 Contents
    page_break(doc)
    add_chapter_title(doc, "Report guide", "Contents", "The report follows the prescribed submission sequence")
    toc_rows = [
        ("Chapter 0", "Abstract of the Project", "3"),
        ("Chapter 1", "Introduction", "4"),
        ("", "Project Context and Architecture", "5"),
        ("Chapter 2", "Problem Statement", "6"),
        ("Chapter 3", "Data Collection", "7"),
        ("", "Dataset Design and Governance", "8"),
        ("Chapter 4", "Methodology", "9"),
        ("", "Feature Engineering and Model Design", "10"),
        ("", "Application and Safety Design", "11"),
        ("Chapter 5", "Experiment and Results", "12"),
        ("", "Comparative Results", "13"),
        ("", "Discussion and Managerial Reading", "14"),
        ("Chapter 6", "Summary and Conclusion", "15"),
        ("Annexure A", "Code Structure and Selected Logic", "16"),
        ("Annexure B", "Dataset Links and References", "17"),
        ("Annexure C", "Project Evidence and Submission Notes", "18"),
    ]
    add_table(doc, ["Section", "Title", "Page"], toc_rows, [1.15, 4.9, .65], font_size=9.2)
    add_body(doc, "Scope note  Smart Sleep is a wellness prototype. It estimates the likelihood of light sleep inside a selected wake window; it does not diagnose sleep disorders or claim clinical sleep staging accuracy.", bold_lead="Scope note", size=9.6, after=0)

    # Page 3 Abstract
    page_break(doc)
    add_chapter_title(doc, "Chapter 0", "Abstract of the Project", "A local-first smart alarm built around wearable signals and a dependable fallback")
    add_body(doc, "Waking at an inconvenient point in the sleep cycle can leave a person feeling unusually groggy even after a reasonable night in bed. Our project explores a practical response to that everyday problem. Smart Sleep Alarm combines a Samsung Galaxy Watch4, an Android phone and a machine-learning pipeline to look for a likely light-sleep opportunity during a wake window selected by the user.")
    add_body(doc, "The watch records accelerometer, heart-rate and inter-beat-interval signals. It stages ordered batches locally before sending them to the phone, which protects the session when Bluetooth drops or either device restarts. The phone stores the observations, converts them into causal 30-second feature epochs and can run an on-device model. A separate exact-time phone alarm remains armed throughout. If sensing, transfer, storage or inference fails, the user is still woken at the target time.")
    add_picture(doc, BUILD / "architecture.png", 6.9, "Figure 1  Smart Sleep Alarm end-to-end operating model")
    add_body(doc, "For model development, we used BIDSleep version 1.0.0, an open dataset containing 253 nights from 47 healthy participants. Participant-level splits prevent the same person from appearing in training and test data. The strongest frozen test F1 in the completed experiments was 0.626 for an improved causal logistic model. The result is useful for a prototype, but the modest ROC AUC of 0.572 and cross-device gap between Apple Watch training data and Samsung Watch deployment limit the claim we can responsibly make.")

    # Page 4 Introduction
    page_break(doc)
    add_chapter_title(doc, "Chapter 1", "Introduction", "Why the wake moment matters and why a smartwatch is a useful place to investigate it")
    add_body(doc, "Most alarms optimise one variable: time. They ring when the clock reaches a fixed target, regardless of whether the sleeper appears restless, awake or deeply asleep. That simplicity makes them dependable, but it also ignores information modern wearables already collect. Smart Sleep Alarm asks whether those signals can support a better-timed wake-up without weakening the basic promise of an alarm.")
    add_subheading(doc, "Project objective")
    add_body(doc, "Our objective was to build and evaluate a working prototype that estimates the probability of light sleep from recent wearable observations, then uses that estimate only within a user-defined wake window. The project had to satisfy three conditions: operate with sensors available on a Galaxy Watch4, keep personal data local by default and preserve an exact-time fallback even when the intelligent path breaks.")
    add_subheading(doc, "Business relevance")
    add_body(doc, "The idea sits at the intersection of consumer wellness, edge analytics and responsible product design. Users often value a simple outcome more than a technically impressive model. Here, the outcome is waking on time and, when the evidence is good enough, waking a little more comfortably. That makes reliability, privacy and honest communication part of the product itself rather than supporting details.")
    add_table(doc, ["Stakeholder", "Primary expectation", "Design response"], [
        ("User", "Wake on time with minimal setup", "Exact-time fallback and short wake window"),
        ("Product team", "A demonstrable end-to-end prototype", "Watch, phone, model and alarm integration"),
        ("Institution", "Evidence-backed analysis", "Reproducible splits, metrics and documented limits"),
        ("Future partner", "Safe handling of health-adjacent data", "Local storage, consent and restrained claims"),
    ], [1.15, 2.45, 3.1], font_size=8.9)

    # Page 5 Context and architecture
    page_break(doc)
    add_chapter_title(doc, "Chapter 1", "Project Context and Architecture", "A two-device design separates sensing from dependable alarm delivery")
    add_body(doc, "We chose a paired Watch and phone architecture because each device is good at a different job. The watch is close to the body and can capture motion and pulse signals. The phone has more dependable audio, durable storage and Android alarm services. Keeping those responsibilities separate also gives the system a clear failure path: if the watch disappears, the phone alarm still has enough information to fire.")
    add_table(doc, ["Layer", "Responsibility", "Implemented mechanism"], [
        ("Wear OS", "Capture and stage sensor data", "Foreground service, private files and bounded batches"),
        ("Transport", "Move ordered observations", "Versioned codec, urgent Data Items and acknowledgements"),
        ("Android", "Persist sessions and own fallback", "Room repository and AlarmManager setAlarmClock"),
        ("Core", "Share domain and feature rules", "Pure Kotlin models, validation and feature pipeline"),
        ("ML", "Prepare, train and evaluate", "Python pipeline with subject-held-out evaluation"),
    ], [1.05, 2.2, 3.45], font_size=9)
    add_subheading(doc, "Operating boundaries")
    add_bullets(doc, [
        "No accounts, cloud sync, advertising or analytics are required for the prototype.",
        "Microphone recording and medical diagnosis remain outside the submitted scope.",
        "Samsung Health sleep stages may support future personal calibration, but they aren't treated as independent clinical ground truth.",
        "The supported hardware scope is Samsung Galaxy Watch4 and later Samsung-powered Wear OS watches.",
    ], size=10)
    add_body(doc, "This architecture is deliberately conservative. It accepts that Bluetooth, sensors and models can fail, and it assigns the non-negotiable wake-up function to the component best placed to deliver it.")

    # Page 6 Problem
    page_break(doc)
    add_chapter_title(doc, "Chapter 2", "Problem Statement", "How can a smart alarm improve timing without becoming less trustworthy than a normal alarm")
    add_body(doc, "The central problem is a decision under uncertainty. A consumer watch does not observe sleep directly. It observes motion and cardiovascular signals that are correlated with sleep state, and those signals vary across people, devices and nights. A useful product must act on imperfect evidence while keeping the cost of a missed alarm close to zero.")
    add_subheading(doc, "Decision question")
    add_body(doc, "Within the interval between the user's earliest acceptable wake time and final target time, should the system trigger now because recent evidence points to light sleep, or should it wait? At the target time, waiting is no longer allowed. The fallback must trigger even if the model has produced no valid estimate.")
    add_table(doc, ["Problem dimension", "Risk", "Project response"], [
        ("Signal quality", "Loose fit, missing pulse or off-body readings", "Quality ratios, nullable values and off-body flags"),
        ("Connectivity", "Watch batches arrive late or more than once", "Durable staging, ordered sequences and deduplication"),
        ("Data leakage", "Model appears stronger because the same person appears twice", "Participant-disjoint train, validation and test splits"),
        ("Class imbalance", "Accuracy hides poor light-sleep recognition", "F1, precision, recall, AUC and confusion matrices"),
        ("Product safety", "Inference failure causes a missed alarm", "Independent exact-time phone fallback"),
        ("Claim quality", "Wellness output is mistaken for diagnosis", "Probability language and an explicit non-medical boundary"),
    ], [1.25, 2.35, 3.1], font_size=8.7)
    add_body(doc, "We defined success as a credible prototype rather than a medical-grade sleep-stage classifier. The technical system had to preserve data order, produce reproducible model evidence and demonstrate safe alarm behaviour. Model performance mattered, but it couldn't be the only measure of success.")

    # Page 7 Data collection
    page_break(doc)
    add_chapter_title(doc, "Chapter 3", "Data Collection", "Public labelled sleep data for training and live wearable data for system validation")
    add_body(doc, "The project uses two distinct data streams. BIDSleep supplies labelled historical data for model development. The Galaxy Watch4 supplies live, unlabelled observations for validating capture, transfer, storage and application behaviour. Keeping those purposes separate avoids presenting live demonstrations as model ground truth.")
    add_table(doc, ["Source", "Signals", "Role", "Scale or status"], [
        ("BIDSleep 1.0.0", "Apple Watch acceleration and heart rate with EEG stages", "Training and held-out evaluation", "47 participants and 253 nights"),
        ("Galaxy Watch4", "Acceleration, heart rate and IBI", "Hardware and workflow testing", "Owner-device prototype sessions"),
        ("Samsung Health", "Consumer sleep-stage summaries", "Possible future calibration", "Not used as training ground truth"),
        ("MESA Sleep", "Actigraphy, PSG and ECG-derived information", "Planned independent comparison", "Separate future pilot"),
    ], [1.2, 2.65, 1.65, 1.35], font_size=8.5)
    add_subheading(doc, "Collection controls")
    add_bullets(doc, [
        "Raw public data stays outside the Git repository; the project records dataset version, mapping, split seed and feature order.",
        "Watch data is written to private device storage before upload and is deleted only after a durable acknowledgement.",
        "Every sensor batch carries a session identifier, sequence number and ordered timestamps.",
        "Unknown sleep labels are counted and removed rather than silently reassigned.",
    ], size=9.9)
    add_body(doc, "BIDSleep was selected because its wearable signals are closer to the product's inputs than PSG-only datasets. The compromise is device mismatch: it was collected with an Apple Watch, while our application runs on Samsung hardware. We treat that mismatch as a material limitation throughout the report.")

    # Page 8 Dataset design
    page_break(doc)
    add_chapter_title(doc, "Chapter 3", "Dataset Design and Governance", "The split follows participants, not individual epochs")
    add_picture(doc, BUILD / "dataset_profile.png", 6.85, "Figure 2  Subject allocation and held-out test composition")
    add_body(doc, "After cleaning and alignment, the pipeline created causal sequences of ten 30-second epochs. The completed run contained 132,355 training sequences, 41,656 validation sequences and 35,740 test sequences. Twenty-nine participants were assigned to training, nine to validation and nine to test with seed 20260821.")
    add_subheading(doc, "Label definition")
    add_body(doc, "N1 and N2 were mapped to light sleep. Wake, N3 and REM were mapped to the other class. This binary label matches the product decision more closely than a five-stage classifier: the alarm needs a cautious indication that a wake opportunity may be acceptable, not a complete clinical interpretation of the night.")
    add_table(doc, ["Governance choice", "Reason"], [
        ("Participant-held-out split", "Prevents personal movement and heart-rate patterns from leaking into the test set"),
        ("Causal windows", "Uses only the current and preceding epochs, matching what the live app can know"),
        ("Versioned manifest", "Makes the release, feature order, labels and split seed auditable"),
        ("No raw data in Git", "Respects storage, licence and privacy boundaries"),
    ], [2.05, 4.65], font_size=9)

    # Page 9 Methodology
    page_break(doc)
    add_chapter_title(doc, "Chapter 4", "Methodology", "From raw wearable measurements to a time-bounded alarm decision")
    add_picture(doc, BUILD / "method_flow.png", 6.9, "Figure 3  Causal processing and decision sequence")
    add_body(doc, "The pipeline begins with timestamped acceleration, heart-rate and IBI samples. Validation rejects malformed records, impossible ordering and unsupported payloads. Samples are then grouped into 30-second epochs, where feature values are calculated only from observations available at that time.")
    add_body(doc, "Ten consecutive epochs form a five-minute context window. That window is long enough to represent short trends but remains practical for on-device inference. The model emits a probability rather than a hard sleep-stage declaration. The decision layer can smooth recent estimates and compare them with a threshold only after the wake window opens.")
    add_subheading(doc, "Research sequence")
    add_table(doc, ["Phase", "Question answered", "Evidence"], [
        ("Data preparation", "Can raw records be converted consistently", "Manifest, checks and subject splits"),
        ("Baseline", "Does a simple model detect useful signal", "Logistic regression test metrics"),
        ("Deep models", "Does temporal complexity improve performance", "CNN GRU and CNN LSTM comparison"),
        ("Improvement", "Can causal context raise F1 without leakage", "Validation selection and frozen test result"),
        ("Application", "Can the result support a safe product flow", "On-device path plus exact-time fallback"),
    ], [1.25, 2.65, 2.8], font_size=8.8)

    # Page 10 Features and models
    page_break(doc)
    add_chapter_title(doc, "Chapter 4", "Feature Engineering and Model Design", "Simple, interpretable features were tested alongside temporal neural networks")
    add_table(doc, ["Feature group", "Examples", "What it may capture"], [
        ("Movement", "Magnitude mean, dispersion, median deviation", "Stillness, movement intensity and variability"),
        ("Activity pattern", "Activity count and zero-crossing rate", "Frequency of direction or activity changes"),
        ("Heart rate", "Mean and standard deviation", "Cardiovascular level and short-term variation"),
        ("IBI", "Mean interval and RMSSD", "Beat timing and short-term variability"),
        ("Quality", "Valid-sample ratios and off-body flag", "Whether the estimate should be trusted"),
        ("Context", "Ten-epoch history and clock encodings", "Recent trend and position within the session"),
    ], [1.2, 2.7, 2.8], font_size=8.7)
    add_subheading(doc, "Candidate models")
    add_body(doc, "The baseline model used logistic regression over causal engineered features. It offered a useful reference because its coefficients are inspectable and its computational cost is small. We then trained CNN-GRU and CNN-LSTM candidates. Their convolutional layers encode local feature patterns, while the recurrent layer carries information across the ten-epoch sequence.")
    add_body(doc, "A raw-signal CNN-GRU diagnostic tested whether less summarised input could recover information lost during feature engineering. The final improvement experiment expanded the causal feature view and selected a regularised logistic model on validation data. The held-out test set remained untouched until the configuration and threshold were fixed.")
    add_table(doc, ["Candidate", "Parameters", "Selection logic"], [
        ("Baseline logistic", "121", "Reference for complexity and interpretability"),
        ("CNN GRU", "7,553", "Temporal neural candidate"),
        ("CNN LSTM", "9,665", "Alternative recurrent memory"),
        ("Improved logistic", "C 0.03", "Best validation F1 among causal feature trials"),
    ], [2.2, 1.3, 3.4], font_size=8.8)

    # Page 11 Application safety
    page_break(doc)
    add_chapter_title(doc, "Chapter 4", "Application and Safety Design", "The intelligent path can fail without cancelling the alarm")
    add_body(doc, "The most important design choice was to separate model confidence from alarm certainty. A sleep estimate may be missing, late or wrong. The exact target time isn't ambiguous. Android's alarm service therefore owns the fallback independently of the watch and inference pipeline.")
    add_table(doc, ["Scenario", "System behaviour", "User outcome"], [
        ("Light-sleep threshold met", "Claim session once and trigger within the window", "Earlier wake opportunity"),
        ("No valid estimate", "Continue waiting while fallback stays armed", "Wake at target time"),
        ("Bluetooth interruption", "Watch retains ordered batches and retries", "Data can recover after reconnection"),
        ("Duplicate delivery", "Sequence key prevents duplicate persistence", "Stable session history"),
        ("Model exception", "Return no estimate and preserve fallback", "No missed alarm from inference failure"),
        ("User dismisses", "Stop sound and record terminal state", "Clear, bounded alarm behaviour"),
    ], [1.5, 3.3, 1.9], font_size=8.8)
    add_subheading(doc, "Interface evidence")
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_repeat_table_header(table.rows[0])
    for cell in table.rows[0].cells:
        set_cell_border(cell, color="FFFFFF", size="0")
        set_cell_margins(cell, 30, 80, 30, 80)
    p1 = table.cell(0, 0).paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p1.add_run().add_picture(str(PHONE_IMAGE), height=Inches(2.45))
    p2 = table.cell(0, 1).paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.add_run().add_picture(str(WATCH_IMAGE), height=Inches(2.45))
    add_caption(doc, "Figure 4  Android alarm setup and Watch sensor controls used during prototype testing")

    # Page 12 Experiment setup
    page_break(doc)
    add_chapter_title(doc, "Chapter 5", "Experiment and Results", "Techniques, evaluation rules and reproducibility controls")
    add_body(doc, "We evaluated the models on participant-held-out BIDSleep sequences. Training used seed 20260821, and the deep-learning run executed on an NVIDIA GeForce RTX 4060 Laptop GPU. Validation data selected hyperparameters and thresholds. Test data was reserved for the frozen comparison.")
    add_table(doc, ["Metric", "Why it was included"], [
        ("Precision", "Shows how often a predicted light-sleep opportunity was actually labelled light sleep"),
        ("Recall", "Shows how many labelled light-sleep epochs the model detected"),
        ("F1 score", "Balances precision and recall when class proportions are uneven"),
        ("ROC AUC", "Measures ranking quality across thresholds"),
        ("Confusion matrix", "Makes false triggers and missed opportunities visible"),
        ("Parameter count", "Keeps deployment cost in view while comparing complexity"),
    ], [1.45, 5.25], font_size=9)
    add_subheading(doc, "Techniques used")
    add_bullets(doc, [
        "Deterministic subject-level splitting and a recorded dataset manifest.",
        "Median filling and scaling fitted on training data only.",
        "Class-aware logistic regression as the tabular baseline.",
        "One-dimensional convolution combined with GRU and LSTM sequence layers.",
        "Validation-selected probability thresholds and frozen test reporting.",
        "ONNX export checks for the selected deployment path in the application prototype.",
    ], size=9.9)
    add_body(doc, "The test set contained 35,740 sequences. Of these, 16,381 were labelled light sleep and 19,359 were labelled as another stage. That balance makes raw accuracy less misleading than in a severely imbalanced dataset, but it still doesn't replace class-specific measures.")

    # Page 13 comparative results
    page_break(doc)
    add_chapter_title(doc, "Chapter 5", "Comparative Results", "The best F1 came from a strengthened simple model, not the deepest network")
    add_picture(doc, BUILD / "model_comparison.png", 6.75, "Figure 5  Held-out BIDSleep comparison across completed candidates")
    add_table(doc, ["Model", "Precision", "Recall", "F1", "ROC AUC"], [
        ("Baseline logistic", "0.500", "0.616", "0.552", "0.573"),
        ("CNN GRU", "0.469", "0.481", "0.475", "0.527"),
        ("CNN LSTM", "0.463", "0.581", "0.515", "0.525"),
        ("Raw CNN GRU", "0.453", "0.999", "0.623", "0.515"),
        ("Improved logistic", "not separately recorded", "not separately recorded", "0.626", "0.572"),
    ], [2.0, 1.2, 1.2, .9, 1.1], font_size=8.6)
    add_body(doc, "The improved logistic model achieved the strongest completed test F1 at 0.626, an absolute gain of 0.074 over the original baseline. Its AUC remained moderate. The raw CNN-GRU reached a similar F1 largely by predicting light sleep almost everywhere; its recall was 0.999, precision 0.453 and AUC 0.515. That behaviour would create many premature wake opportunities and is not a persuasive deployment result.")

    # Page 14 discussion
    page_break(doc)
    add_chapter_title(doc, "Chapter 5", "Discussion and Managerial Reading", "Model scores matter, but the error pattern determines whether the product feels responsible")
    add_picture(doc, BUILD / "confusion_matrices.png", 6.9, "Figure 6  Confusion matrices for the original baseline and deep sequence candidates")
    add_body(doc, "The first lesson is that additional model complexity didn't automatically improve generalisation. Both engineered-feature neural models fell below the baseline F1 and AUC on the held-out participants. The stronger simple model benefited from a better causal representation and validation discipline rather than a larger parameter count.")
    add_body(doc, "The second lesson concerns incentives. A model can raise F1 by calling many epochs light sleep. The raw CNN-GRU demonstrates the danger: near-perfect recall looks attractive until its low precision and near-random AUC are considered. For an alarm, false positives have a visible cost because they may wake the user earlier than necessary. Threshold choice should therefore reflect the wake-window length and tolerance for early triggers, not a single headline score.")
    add_subheading(doc, "What the evidence supports")
    add_bullets(doc, [
        "The pipeline contains measurable signal above a trivial random ranking, but current discrimination is modest.",
        "Causal feature design and leakage control were more valuable than network depth in this experiment.",
        "The prototype is suitable for classroom demonstration and further field testing, not clinical or commercial performance claims.",
        "Personal calibration and an independent Samsung-compatible dataset are logical next tests.",
    ], size=9.8)

    # Page 15 conclusion
    page_break(doc)
    add_chapter_title(doc, "Chapter 6", "Summary and Conclusion", "A credible smart alarm starts with dependable behaviour, then earns the right to be intelligent")
    add_body(doc, "Smart Sleep Alarm demonstrates a complete line of reasoning from consumer problem to operating prototype. The Watch captures movement and cardiovascular signals, the transport layer preserves ordered batches, the phone stores the session and owns an exact-time fallback, and the ML pipeline estimates a possible light-sleep opportunity from causal context.")
    add_body(doc, "The completed BIDSleep experiments also produced a useful management lesson. The simplest improved candidate delivered the highest frozen test F1 of 0.626. More complex neural models did not generalise as well, and one raw-signal model achieved a high F1 through an impractical tendency to predict light sleep. Looking at the error pattern changed the interpretation of the score.")
    add_subheading(doc, "Project contribution")
    add_table(doc, ["Area", "Contribution"], [
        ("Product", "Defined a narrow wake-window experience with a non-negotiable fallback"),
        ("Engineering", "Built watch capture, resilient transfer, phone persistence and alarm control"),
        ("Analytics", "Created a reproducible causal modelling and comparison pipeline"),
        ("Governance", "Kept data local and separated wellness estimates from medical claims"),
    ], [1.45, 5.25], font_size=9.2)
    add_body(doc, "Our conclusion is measured. The project shows that wearable sensing can support a thoughtful smart-alarm prototype, but the current model does not justify a promise that every wake-up will occur during light sleep. The next stage should combine consented overnight field trials, calibration by user and a broader cross-device evaluation. Until that evidence exists, the fallback alarm remains the product's most important feature.")

    # Page 16 Code annexure
    page_break(doc)
    add_chapter_title(doc, "Annexure A", "Code Structure and Selected Logic", "Key modules are separated by device, domain responsibility and modelling task")
    add_table(doc, ["Path", "Purpose"], [
        ("app/src/main", "Phone UI, Room storage, transport receiver and fallback alarm"),
        ("wear/src/main", "Sensor capture, batch staging, Data Layer transfer and haptics"),
        ("core/src/main", "Shared session contracts, codec and feature calculations"),
        ("ml/prepare_dataset.py", "BIDSleep cleaning, epoch preparation and split artefacts"),
        ("ml/train.py", "Baseline, CNN-GRU and CNN-LSTM training"),
        ("ml/evaluate.py", "Frozen model comparison and metric generation"),
        ("ml/export_onnx.py", "Portable model export and parity verification"),
    ], [2.3, 4.4], font_size=8.8)
    add_subheading(doc, "Representative decision logic")
    code = (
        "if now >= target_time:\n"
        "    return TRIGGER_FALLBACK\n"
        "if now < wake_window_start:\n"
        "    return WAIT\n"
        "if two_recent_valid_scores_meet_threshold:\n"
        "    return TRIGGER_LIGHT_SLEEP\n"
        "return WAIT"
    )
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(.35)
    p.paragraph_format.right_indent = Inches(.35)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    p_pr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F1F3F5")
    p_pr.append(shd)
    set_run_font(p.add_run(code), name="Consolas", size=9.2)
    add_body(doc, "Repository  https://github.com/cs-amit/ABA-Project", bold_lead="Repository", size=10)
    add_body(doc, "The repository excludes raw health datasets and the Samsung proprietary SDK binary. Tests cover batch validation, persistence, fallback claiming, model data contracts and export behaviour.", size=10)

    # Page 17 Data and references
    page_break(doc)
    add_chapter_title(doc, "Annexure B", "Dataset Links and References", "Public sources are linked because raw data is not bundled with the submission")
    add_table(doc, ["Resource", "Link and use"], [
        ("BIDSleep 1.0.0", "https://physionet.org/content/bidsleep-dataset/1.0.0/  Primary model dataset"),
        ("MESA Sleep", "https://sleepdata.org/datasets/mesa  Planned separate comparison"),
        ("Project repository", "https://github.com/cs-amit/ABA-Project  Source code and documentation"),
        ("Samsung Health Sensor SDK", "https://developer.samsung.com/health/sensor  Device integration reference"),
        ("Android exact alarms", "https://developer.android.com/develop/background-work/services/alarms  Fallback implementation reference"),
    ], [1.75, 4.95], font_size=8.7)
    add_subheading(doc, "Selected references")
    refs = [
        "Song, T.-A. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels. PhysioNet, version 1.0.0, 2026.",
        "National Sleep Research Resource. MESA Sleep dataset and documentation.",
        "Samsung Developer. Samsung Health Sensor SDK programming guide and tracker documentation.",
        "Android Developers. Schedule alarms and AlarmManager documentation.",
        "Project model card, dataset manifest, training run and source tests in the ABA-Project repository.",
    ]
    for idx, ref in enumerate(refs, 1):
        add_body(doc, f"{idx}. {ref}", size=9.7, after=5, align=WD_ALIGN_PARAGRAPH.LEFT)
    add_subheading(doc, "Reproduction note")
    add_body(doc, "Raw BIDSleep files should be downloaded from PhysioNet under its published terms and stored outside the repository. The preparation pipeline writes epochs.parquet, splits.json and dataset_manifest.json. Those artefacts preserve the release, label mapping, subject allocation and feature order used in the experiments.", size=9.9)

    # Page 18 evidence and submission notes
    page_break(doc)
    add_chapter_title(doc, "Annexure C", "Project Evidence and Submission Notes", "A concise map of the material available for review and viva discussion")
    add_table(doc, ["Evidence", "What it demonstrates", "Location"], [
        ("Android and Wear OS modules", "Sensor capture, durable transfer, persistence and fallback alarm control", "app, wear and core"),
        ("Automated tests", "Validation of codecs, features, alarms, storage and model contracts", "Module test folders"),
        ("Dataset records", "BIDSleep release, label mapping, participant split and feature order", "ML artefacts"),
        ("Model results", "Candidate metrics, confusion matrices and selection evidence", "ML evaluation outputs"),
        ("Project documents", "Design choices, data protocol, model limits and implementation status", "docs folder"),
    ], [1.65, 3.55, 1.5], font_size=8.9)
    add_subheading(doc, "Submission boundaries")
    add_bullets(doc, [
        "Raw public health data and proprietary Samsung SDK binaries are not included.",
        "The model output is a wellness estimate and does not diagnose a sleep stage or disorder.",
        "The exact-time phone alarm remains active regardless of the watch or model state.",
        "The reported results apply to the recorded BIDSleep split and should not be generalised to every user or device.",
    ], size=9.9)
    add_subheading(doc, "Closing statement")
    add_body(doc, "The team approached Smart Sleep Alarm as both a product problem and an evidence problem. We built the end-to-end path, recorded where it worked, and kept the claims narrower than the ambition. That discipline leaves the project in a useful position: the prototype can be demonstrated today, while the next round of data can be used to decide whether its smart timing genuinely improves the user's morning.", size=10.5)
    add_body(doc, "Submitted to Indian Institute of Management Jammu", bold_lead="Submitted to", size=10.2, after=0, align=WD_ALIGN_PARAGRAPH.CENTER)

    props = doc.core_properties
    props.title = "Smart Sleep Alarm Project Report"
    props.subject = "MBA project report on wearable sensing and smart alarm decision design"
    props.author = "Smart Sleep Alarm Project Team"
    props.keywords = "Smart Sleep Alarm, IIM Jammu, Wear OS, BIDSleep, machine learning"
    alt_texts = [
        "Abstract wearable signal approaching a highlighted wake window and safe alarm decision",
        "Flow diagram from Galaxy Watch4 sensors through the phone and model to the wake decision",
        "Charts showing the participant split and labelled composition of the held-out test epochs",
        "Six-step flow from sensor capture and cleaning to the final alarm decision",
        "Android phone screen showing Smart Sleep alarm notification setup",
        "Galaxy Watch screen with start sensor, stop sensor and dismiss alarm controls",
        "Bar chart comparing held-out F1 and ROC AUC across five model candidates",
        "Confusion matrices for the baseline logistic, CNN GRU and CNN LSTM models",
    ]
    for inline_shape, description in zip(doc.inline_shapes, alt_texts):
        inline_shape._inline.docPr.set("descr", description)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    create_document()
