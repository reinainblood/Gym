# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upstream control: the verdict-parsing helpers of the official FACTS Grounding v2 starter notebook.

Copied verbatim (Apache 2.0) from Kaggle notebook ``prathameshbang/facts-grounding-v2-benchmark-starter`` version 4:
``extract_instruction_json`` (eligibility verdict), ``clean_json_content_8`` and ``parse_ufg_rev21_verdict``
(sentence-level grounding verdict). The SHA-256 values pin the source cells. ``calibrate.py`` replays every judge
receipt through these functions and compares the result with the Gym verifier's decisions.
"""

import json
import re


STARTER_NOTEBOOK = "prathameshbang/facts-grounding-v2-benchmark-starter"
STARTER_VERSION = 4
QUALITY_CELL_SHA256 = "71f028c10a4630b82acdf44217a54f51c489f2a0be94340ad10fd475bb00669f"  # pragma: allowlist secret
GROUNDING_CELL_SHA256 = "a15d65cf94da9ad00558cb5bc0f9ef35c642eb1bf46f5d893297ee0fef18a48e"  # pragma: allowlist secret


def extract_instruction_json(ans: str) -> dict:
    pattern = re.compile(
        r"{\s*"
        r'"Instruction Following"\s*:\s*'
        r'"(No Issues|Minor Issue\(s\)|Major Issue\(s\))"\s*'
        r"}"
    )

    match = pattern.search(ans)

    if match:
        json_string = match.group(0)
        try:
            # Once the valid JSON string is found, parse it
            return json.loads(json_string)
        except Exception as e:
            # This is unlikely to happen if the regex matches, but it's good practice
            print("!!Bad Json!!")
            print(e)
            print(ans)
            return {"Instruction Following": "Invalid"}

    return {"Instruction Following": "Invalid"}


def clean_json_content_8(text: str, start_field: str, end_field: str | None = None) -> str:
    r"""Cleans the value of a specified field in a JSON-like string.

    This function can clean a field in the middle of a JSON object or the
    very last field. The output of a model might be a string like:

    {
      "sentence": "The first sentence.",
      "rationale": "The first rationale.",
      "excerpt": "  \\"The first excerpt.\\"  ",
      "label": "supported"
    }

    This function can target a specific field's value (e.g., "excerpt"),
    remove extra quotes and escape characters, and reconstruct the string.

    Args:
      text: The full JSON-like string to clean.
      start_field: The name of the field whose value needs to be cleaned.
      end_field: The name of the field that immediately follows the start_field.
        If start_field is the last field in the object, this should be set to
        None.

    Returns:
      The cleaned string if the specified field is found, otherwise the
      original string.
    """

    # The part of the regex that looks for what comes *after* the value.
    # It's either the next field or the closing brace of the JSON object.
    if end_field:
        # Pattern for a field followed by another field.
        # Captures the comma and the next field key to preserve it.
        trailer_pattern = rf'(,\s*"{re.escape(end_field)}":)'
    else:
        # Pattern for the last field in the object.
        # Captures optional whitespace and the closing brace to preserve it.
        trailer_pattern = r"(\s*})"

    pattern = re.compile(rf'"{re.escape(start_field)}":(.+?){trailer_pattern}', re.DOTALL)

    def _clean_match(match: re.Match[str]) -> str:
        content = match.group(1)
        trailer = match.group(2)

        content = content.replace('"', "")
        content = content.replace("\\", "")
        content = content.replace("\x02", "")
        content = content.strip()

        cleaned_content = f'"{content}"' if content != "null" else "null"
        return f'"{start_field}":{cleaned_content}{trailer}'

    cleaned_text, num_subs = pattern.subn(_clean_match, text, count=1)
    return cleaned_text if num_subs > 0 else text


def parse_ufg_rev21_verdict(answer: str, judge: str, verbose: bool = False) -> tuple[bool, list[dict[str, str]], bool]:
    """Parses a structured JSON answer into a boolean grounding prediction."""
    if "```json" in answer:
        jsonl_chunks = []
        for a in answer.split("```json")[1:]:
            jsonl_chunks.append(a.split("```")[0])
        answer = "\n".join(jsonl_chunks)
    answer = answer.strip()
    answer = answer.replace("}\n", "}\n@\n@\n")
    answer = answer.replace("} \n", "}\n@\n@\n")
    answer = answer.replace("}  \n", "}\n@\n@\n")
    answer = answer.replace("<ctrl75>", "")
    parsed_answers = []
    for line in answer.split("\n@\n@\n"):
        raw_line = line
        try:
            line = line.replace("\n", " ")
            line = line.replace("\\'", "'")
            line = line.replace("<ctrl75>", "")
            # pylint-enable: g-inconsistent-quotes
            line = line.lstrip(",")
            #  Remove any additional newlines that manifest
            #  as a result of the above replacements.
            line = line.replace("\n", " ")
            line = clean_json_content_8(line, "sentence", "label")
            line = clean_json_content_8(line, "label", "rationale")
            line = clean_json_content_8(line, "rationale", "excerpt")
            line = clean_json_content_8(line, "excerpt")
            parsed = json.loads(line)
            if "label" not in parsed:
                parsed["label"] = "unknown"
            parsed_answers.append(parsed)
        except (json.JSONDecodeError, ValueError, TypeError):
            if verbose:
                print(
                    f"error with parsing sentence from {judge} model response to json. The full"
                    f" raw line is: {raw_line}"
                )

            if '"not_supported"' in raw_line:
                parsed_answers.append(
                    {
                        "sentence": "",
                        "rationale": "",
                        "excerpt": "",
                        "label": "not_supported",
                    }
                )
    if parsed_answers:
        bool_ans = all(
            parsed_answer["label"] in ("supported", "no_rad", "unknown") for parsed_answer in parsed_answers
        )
    else:
        if verbose:
            print(f"error with parsing json response from {judge} model. The full answer is: {answer}")
        bool_ans = False
    # Only returns unsupported and contradictory sentences to use for rationales.
    filtered_parsed_answers = [
        parsed_answer for parsed_answer in parsed_answers if parsed_answer["label"] in ["not_supported"]
    ]

    return bool_ans, filtered_parsed_answers
