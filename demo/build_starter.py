"""Build a genuine, pretrained GeCo starter project for the hosted sandbox.

Run once at image build time. Visitors receive independent copies of the
resulting project; no embeddings or training are needed merely to start.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from geometric_coder import GeometricCoder
from geometric_coder.recipes import raw_tfidf, semantic_fast


TOPICS = {
    "Community participation": [
        "Residents were invited to shape the neighborhood planning process.",
        "The town hall allowed families to question the proposed changes.",
        "Local volunteers organized a forum for community feedback.",
        "Parents described being excluded from school decision making.",
        "Public meetings gave residents a direct voice in local policy.",
        "The committee consulted community members before its vote.",
        "A resident asked whether the council would listen to dissent.",
        "Organizers held listening sessions across the neighborhood.",
        "Participants disagreed over who should be represented.",
        "Residents co-designed a new neighborhood garden.",
        "The advisory group opened a public comment period.",
        "Several residents felt their concerns had been ignored.",
    ],
    "Resources and inequality": [
        "Families reported difficulty finding affordable housing.",
        "The school lacked books and adequate computer equipment.",
        "Public transportation costs prevented some residents from attending.",
        "Wealthier districts had more experienced teachers.",
        "The scholarship helped students afford college tuition.",
        "Workers struggled with rising rent and stagnant wages.",
        "Rural households experienced unreliable internet access.",
        "Families shared food when benefits were delayed.",
        "The report compared resources across neighborhoods.",
        "Some patients could not afford their prescriptions.",
        "The district provided laptops to low-income students.",
        "The clinic expanded its sliding-scale payment program.",
    ],
    "Institutional trust": [
        "Residents questioned whether the agency would keep its promise.",
        "The principal explained the decision transparently to parents.",
        "Families expressed confidence in the new school leadership.",
        "Staff criticized the administration for withholding information.",
        "The official acknowledged earlier mistakes at the meeting.",
        "Community members doubted the accuracy of the public report.",
        "The review committee published detailed meeting minutes.",
        "Residents described the office as fair and responsive.",
        "The delayed response weakened confidence in the institution.",
        "A student praised the university's clear grievance procedure.",
        "The agency released data to improve public accountability.",
        "Residents expected an independent review of the decision.",
    ],
}


def main() -> None:
    root = Path(os.getenv("GECO_DEMO_TEMPLATE", "/app/demo_data/starter.geco"))
    rows = [
        {"document_id": group + 1, "sentence_id": i + 1,
         "text": text, "topic": topic, "source": f"Example {group + 1}"}
        for group, (topic, texts) in enumerate(TOPICS.items())
        for i, text in enumerate(texts)
    ]
    geometries = {"lexical": raw_tfidf()}
    if os.getenv("GECO_DEMO_SEMANTIC", "1") == "1":
        geometries["semantic"] = semantic_fast()
    project = GeometricCoder.create(
        project_dir=root,
        data=pd.DataFrame(rows),
        keys=["document_id", "sentence_id"],
        text="text", modality="text",
        metadata=["topic", "source"],
        geometries=geometries,
        overwrite=True,
    )
    for topic in ("Community participation", "Institutional trust"):
        code_id = project.create_code(
            topic, f"Observation provides evidence concerning {topic.lower()}."
        )
        for row, unit in zip(rows, project.units()):
            label = "positive" if row["topic"] == topic else "negative"
            project.annotate(int(unit["observation_id"]), code_id, label)
        specs = project.classifier_specs(code_id=code_id)
        for spec in specs[:1]:
            project.train_classifier(
                code_id=code_id, classifier_spec_id=int(spec["classifier_spec_id"]),
                tune=False,
            )
    print(f"Built real GeCo demonstration project at {root} with {len(rows)} observations")


if __name__ == "__main__":
    main()
