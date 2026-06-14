EP_FCHD_CLASS_NAMES = [
    "3VT_abnormal",
    "3VT_Norm",
    "A4C_abnormal",
    "A4C_Norm",
]


EP_FCHD_CLASS_DESCRIPTIONS = {
    "3VT_abnormal": (
        "In the anomalous three-vessel view, the two protruding vessels have "
        "a non-V-shaped structure with an indeterminate direction of blood flow."
    ),
    "3VT_Norm": (
        "In a normal three-vessel view, two prominent vessels are displayed in "
        "a V-shaped structure, with blood flow in the same direction when present."
    ),
    "A4C_abnormal": (
        "In the anomalous four-chambered cardiac sectional view, the two prominent "
        "vessels are in a nonparallel configuration with an indeterminate direction "
        "of blood flow."
    ),
    "A4C_Norm": (
        "In a normal four-chamber heart view, two prominent vessels are displayed "
        "in a parallel structure, with consistent blood flow direction."
    ),
}


DEFAULT_UNLABELED_TEXT = (
    "This fetal cardiac ultrasound image may show one of the following findings: "
    "3VT_abnormal, 3VT_Norm, A4C_abnormal, or A4C_Norm."
)
