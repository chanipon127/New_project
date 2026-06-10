# =========================================================
# IMPORT
# =========================================================

import re
import json
import difflib
import requests
import torch
import unicodedata
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding
)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification
)

from sentence_transformers import SentenceTransformer, util

from pythainlp.tokenize import word_tokenize, sent_tokenize
from pythainlp.corpus.common import thai_words
from pythainlp.tag import pos_tag
from pythainlp.util import Trie, normalize

#============== unzip files in dataset =============

import os
# import zipfile 

# base_dir = r"C:\Users\User\Documents\New_project\all_dataset"

# for root, dirs, files in os.walk(base_dir):
#     for file in files:
#         if file.endswith(".zip"):

#             zip_path = os.path.join(root, file)

#             # ชื่อโฟลเดอร์ปลายทาง = ชื่อไฟล์ zip (ตัด .zip ออก)
#             extract_dir = os.path.join(
#                 root,
#                 os.path.splitext(file)[0]
#             )

#             print(f"Extracting: {zip_path}")

#             os.makedirs(extract_dir, exist_ok=True)

#             with zipfile.ZipFile(zip_path, "r") as zip_ref:
#                 zip_ref.extractall(extract_dir)

#             print(f" -> {extract_dir}")

print("Done.")

for root, dirs, files in os.walk(
    r"C:\Users\User\Documents\New_project\all_dataset"
):
    if "config.json" in files:
        print(root)

# =========================================================
# CONFIG
# =========================================================

SEG_MODEL_PATH = r"C:\Users\User\Documents\New_project\all_dataset\S1\2final_model_bi\2final_model_bi"
MAIN_IDEA_MODEL_PATH = r"C:\Users\User\Documents\New_project\all_dataset\S1\4main_idea_model\4main_idea_model_0.68"
S2_MODEL_PATH = r"C:\Users\User\Documents\New_project\all_dataset\S2\S2_scoring_model_weight\S2_scoring_model_weight"

S3_ABBR_CSV_PATH = r"C:\Users\User\Documents\New_project\all_dataset\S3\Abbreviation_last.csv"
S3_CONFIG_PATH = r"C:\Users\User\Documents\New_project\all_dataset\S3\s3_R2_last.csv"
PARAPHRASE_THRESHOLD = 0.5

# หาการยกข้อความ
COPY_THRESHOLD       = 0.90   # similarity > นี้ → ถือว่ายกข้อความ

# S4 paths
S4_MISSPELL_JSON  = r"C:\Users\User\Documents\New_project\all_dataset\S4\update_common_misspellings.json"
S4_LOANWORDS_JSON = r"C:\Users\User\Documents\New_project\all_dataset\S4\thai_loanwords_new_update.json"
S4_KEYWORDS_CSV   = r"C:\Users\User\Documents\New_project\all_dataset\S4\keyword_list_S4_S11_new.csv"
S4_LONGDO_API_KEY    = "d114e6a8d1c3250024a1b0060355f8ea"           # ← ใส่ API key
S4_LONGDO_API_URL    = "https://api.longdo.com/spell-checker/proof"

# S5 path
S5_MODEL_PATH = r"C:\Users\User\Documents\New_project\all_dataset\S5\model_train_S5\model"
S5_MAX_LENGTH        = 416

# S6 path
S6_MODEL_PATH = r"C:\Users\User\Documents\New_project\all_dataset\S6\model_train_S6\model"
S6_MAX_LENGTH        = 416

MAX_LENGTH_SEG  = 512
MAX_LENGTH_MAIN = 256

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# LOAD SEGMENTATION MODEL
# =========================================================

print("Loading Segmentation Model...")

seg_tokenizer = AutoTokenizer.from_pretrained(
    SEG_MODEL_PATH,
    use_fast=True
)

seg_model = AutoModelForTokenClassification.from_pretrained(
    SEG_MODEL_PATH
)

seg_model.to(device)
seg_model.eval()

# =========================================================
# LOAD MAIN IDEA MODEL
# =========================================================

print("Loading Main Idea Model...")

main_tokenizer = AutoTokenizer.from_pretrained(
    MAIN_IDEA_MODEL_PATH,
    use_fast=False
)

main_model = AutoModelForSequenceClassification.from_pretrained(
    MAIN_IDEA_MODEL_PATH
)

main_model.to(device)
main_model.eval()

# =========================================================
# LOAD S2 SCORING MODEL
# =========================================================

print("Loading S2 Scoring Model...")

s2_tokenizer = AutoTokenizer.from_pretrained(S2_MODEL_PATH)

s2_model = AutoModelForSequenceClassification.from_pretrained(
    S2_MODEL_PATH
)

s2_model.to(device)
s2_model.eval()

id2score = {0: 0, 1: 1, 2: 2}

# =========================================================
# LOAD S3 EMBEDDING MODEL
# =========================================================

print("Loading S3 Embedding Model...")

embed_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# =========================================================
# LABEL MAP
# =========================================================

LABEL2ID = seg_model.config.label2id
ID2LABEL = seg_model.config.id2label

B_LABEL_ID = LABEL2ID["B"]

LABEL_COLUMNS = [
    "MAIN_IDEA_1",
    "MAIN_IDEA_2",
    "MAIN_IDEA_3",
    "MAIN_IDEA_4"
]

print("Segmentation Labels:", ID2LABEL)

# =========================================================
# S3 — SETUP (โหลดครั้งเดียว)
# =========================================================

def load_informal_abbr(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return dict(zip(
        df["abbr"].astype(str).str.strip(),
        df["full_words"].astype(str).str.strip()
    ))


def load_config_from_csv(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    required = [
        "allowed_abbr", "reference_text", "title",
        "local_word", "personal pronoun 1", "personal pronoun 2"
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"ไม่พบคอลัมน์ '{col}'")

    allowed_abbr    = set(df["allowed_abbr"].dropna().astype(str).str.strip())
    reference_text  = df["reference_text"].dropna().iloc[0].strip()
    forbidden_list  = df["title"].dropna().astype(str).str.strip().tolist()
    example_phrases = df["local_word"].dropna().astype(str).str.strip().tolist()
    pronouns_all    = (
        df["personal pronoun 1"].dropna().astype(str).tolist() +
        df["personal pronoun 2"].dropna().astype(str).tolist()
    )

    return reference_text, allowed_abbr, forbidden_list, example_phrases, pronouns_all


def create_custom_tokenizer(abbr_map):
    custom_words = set(thai_words())
    custom_words.update(abbr_map.keys())
    return Trie(custom_words)


abbr_map    = load_informal_abbr(S3_ABBR_CSV_PATH)
custom_dict = create_custom_tokenizer(abbr_map)
misspell_dict = {}   # เพิ่มคำผิดได้ที่นี่

(s3_reference_text,
 s3_allowed_abbr,
 s3_forbidden_list,
 s3_example_phrases,
 s3_pronouns_all) = load_config_from_csv(S3_CONFIG_PATH)

# ใช้ reference_text เดียวกันสำหรับตรวจการยกข้อความ
copy_reference_text = s3_reference_text


# =========================================================
# S3 — HELPER FUNCTIONS
# =========================================================

def normalize_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(?<=[.])(?=[ก-ฮA-Z])', ' ', text)
    return text.strip()


def fix_and_collect_typos(text, misspell_dict):
    found      = []
    used_spans = []

    for wrong, right in sorted(
        misspell_dict.items(), key=lambda x: -len(x[0])
    ):
        for match in re.finditer(re.escape(wrong), text):
            span = match.span()
            if any(
                not (span[1] <= s[0] or span[0] >= s[1])
                for s in used_spans
            ):
                continue
            found.append({"wrong": wrong, "right": right})
            used_spans.append(span)

    for e in found:
        text = text.replace(e["wrong"], e["right"])

    return text, found


def check_examples(student_answer, example_phrases):
    found = []
    text  = student_answer
    text  = re.sub(r'เช่น\s*กัน', '', text)
    text  = re.sub(r'เช่น\s*เดียวกัน', '', text)

    for phrase in example_phrases:
        if phrase in text:
            found.append(phrase)

    if re.search(r'เช่น', text):
        found.append("เช่น")

    return list(set(found))


def check_pronouns(student_answer, pronouns_all):
    words = word_tokenize(
        student_answer, engine='newmm', custom_dict=custom_dict
    )
    return [p for p in pronouns_all if p in words]


def check_abbreviations(student_answer, allowed_abbr, abbr_map):
    pattern_dot  = r'(?:[ก-ฮA-Za-z]\.){2,}'
    found_dot    = re.findall(pattern_dot, student_answer)
    words        = word_tokenize(
        student_answer, engine="newmm", custom_dict=custom_dict
    )
    informal_found = list(set([w for w in words if w in abbr_map]))
    invalid_abbr   = []
    valid_abbr     = []

    for abbr in set(found_dot):
        if abbr in allowed_abbr:
            valid_abbr.append(abbr)
        else:
            invalid_abbr.append(abbr)

    return {
        "invalid_abbr":  sorted(invalid_abbr),
        "informal_abbr": informal_found,
        "valid_abbr":    sorted(valid_abbr)
    }


def check_title(student_answer, forbidden_list):
    lines = student_answer.strip().split("\n")

    if len(lines) > 1:
        first_line = lines[0].strip()
        for forbidden in forbidden_list:
            if first_line == forbidden:
                return f"{forbidden} (เป็นชื่อเรื่องแยกบรรทัด)"

    for forbidden in forbidden_list:
        pattern = (
            r'เรื่อง\s*["\u201c\u201d]?' +
            re.escape(forbidden) +
            r'["\u201c\u201d]?'
        )
        if re.search(pattern, student_answer):
            return f"{forbidden} (มีคำว่า 'เรื่อง')"

    for forbidden in forbidden_list:
        pattern = (
            r'^["\u201c\u201d]' +
            re.escape(forbidden) +
            r'["\u201c\u201d][^\w]*'
        )
        if re.search(pattern, student_answer.strip()):
            return f"{forbidden} (เป็นชื่อเรื่องในเครื่องหมายคำพูด)"

    return None


def check_quotes(student_answer):
    quote_pairs = [('"', '"'), ('\u201c', '\u201d'), ("'", "'")]
    issues      = []

    for left, right in quote_pairs:
        if student_answer.count(left) != student_answer.count(right):
            issues.append(f"เครื่องหมาย {left}{right} ไม่สมดุล")

    found_quotes = re.findall(r'["\u201c\u201d\']', student_answer)
    return issues, len(found_quotes)


def check_paraphrase(student_answer, reference_text):
    emb_student   = embed_model.encode(
        student_answer, convert_to_tensor=True
    )
    emb_reference = embed_model.encode(
        reference_text, convert_to_tensor=True
    )
    similarity = float(
        util.cos_sim(emb_student, emb_reference).item()
    )
    is_error = similarity < PARAPHRASE_THRESHOLD
    return is_error, round(similarity, 4)

# =========================================================
# S3 — PREDICT
# =========================================================

def predict_s3_score(text):
    """
    คืนค่า dict:
      S3_SCORE      : 0 หรือ 1
      S3_SIMILARITY : ค่า cosine similarity กับ reference
      S3_REASON     : รายการข้อผิดพลาด (str)
    """

    text = normalize_text(str(text))
    text, _ = fix_and_collect_typos(text, misspell_dict)

    errors = []

    # 1) ตัวอย่าง
    found_examples = check_examples(text, s3_example_phrases)
    if found_examples:
        errors.append(f"พบการยกตัวอย่าง: {', '.join(found_examples)}")

    # 2) สรรพนาม
    found_pronouns = check_pronouns(text, s3_pronouns_all)
    if found_pronouns:
        errors.append(f"พบคำสรรพนาม: {', '.join(found_pronouns)}")

    # 3) คำย่อ
    abbr_result = check_abbreviations(text, s3_allowed_abbr, abbr_map)
    if abbr_result["invalid_abbr"]:
        errors.append(
            f"พบคำย่อไม่อนุญาต: {', '.join(abbr_result['invalid_abbr'])}"
        )
    if abbr_result["informal_abbr"]:
        errors.append(
            f"พบคำไม่เป็นทางการ: {', '.join(abbr_result['informal_abbr'])}"
        )

    # 4) ชื่อเรื่อง
    found_title = check_title(text, s3_forbidden_list)
    if found_title:
        errors.append(f"พบคำต้องห้ามในหัวข้อ: {found_title}")

    # 5) เครื่องหมายคำพูด
    quote_issues, quote_count = check_quotes(text)
    if quote_issues:
        errors.extend(quote_issues)
    if quote_count > 0:
        errors.append("ไม่ควรใช้เครื่องหมายคำพูด")

    # 6) paraphrase
    paraphrase_error, similarity = check_paraphrase(
        text, s3_reference_text
    )
    if paraphrase_error:
        errors.append(
            f"ย่อความไม่เพียงพอ "
            f"(similarity={similarity} < {PARAPHRASE_THRESHOLD})"
        )

    score  = 1 if len(errors) == 0 else 0
    reason = " | ".join(errors) if errors else "ผ่านทุกเงื่อนไข"

    return {
        "S3_SCORE":      score,
        "S3_SIMILARITY": similarity,
        "S3_REASON":     reason
    }

# =========================================================
# S4 — SETUP (โหลดครั้งเดียว)
# =========================================================

print("Loading S4 resources...")

with open(S4_MISSPELL_JSON, 'r', encoding='utf-8') as f:
    _misspell_data = json.load(f)

s4_misspell_dict = {
    item['wrong']: item['right']
    for item in _misspell_data
    if item['right']
}

s4_misspell_whitelist = set(
    item['right'] for item in _misspell_data if item['right']
)

with open(S4_LOANWORDS_JSON, 'r', encoding='utf-8') as f:
    _loanwords_data = json.load(f)

s4_loanwords_whitelist = set(
    item['thai_word'] for item in _loanwords_data
)

_keywords_df = pd.read_csv(S4_KEYWORDS_CSV)

s4_splitable_phrases    = _keywords_df['splitable_phrases'].dropna().tolist()
s4_strict_not_split     = _keywords_df['strict_not_split_word'].dropna().tolist()
s4_allow_list           = _keywords_df['allow_list'].dropna().tolist()
s4_forbid_list          = _keywords_df['forbid_list'].dropna().tolist()
s4_repeatable_words     = _keywords_df['repeatable_words'].dropna().tolist()

s4_thai_dict = set(
    w for w in set(thai_words()) if (' ' not in w) and w.strip()
)

s4_allowed_punctuations = {
    '.', ',', '-', '(', ')', '!', '?', '%',
    '"', '\u201c', '\u201d', '\u2018', '\u2019', '"', "'", '\u2026', 'ๆ', 'ฯ'
}

# =========================================================
# S4 — HELPER FUNCTIONS
# =========================================================

def s4_normalize_thai_variants(text):
    text = re.sub(r'เเ', 'แ', text)
    text = re.sub(r'ํา', 'ำ', text)
    return text


def s4_check_linebreak_issue(prev_tokens, next_tokens, max_words=3):
    last_word  = prev_tokens[-1]
    first_word = next_tokens[0]

    if last_word.endswith('-') or first_word.startswith('-'):
        return False, None, None, None

    for prev_n in range(1, min(max_words, len(prev_tokens)) + 1):
        prev_part = ''.join(prev_tokens[-prev_n:])
        for next_n in range(1, min(max_words, len(next_tokens)) + 1):
            next_part = ''.join(next_tokens[:next_n])
            combined  = normalize(prev_part + next_part)
            if (
                ' ' not in combined
                and combined not in s4_splitable_phrases
                and (
                    combined in s4_strict_not_split
                    or (
                        combined in s4_thai_dict
                        and len(word_tokenize(combined, engine='newmm')) == 1
                    )
                )
            ):
                return True, prev_part, next_part, combined

    return False, None, None, None


def s4_analyze_linebreak_issues(text):
    lines  = text.strip().splitlines()
    issues = []

    for i in range(len(lines) - 1):
        prev_line   = lines[i].strip()
        next_line   = lines[i + 1].strip()
        prev_tokens = word_tokenize(prev_line)
        next_tokens = word_tokenize(next_line)

        if not prev_tokens or not next_tokens:
            continue

        issue, prev_part, next_part, combined = s4_check_linebreak_issue(
            prev_tokens, next_tokens
        )
        if issue:
            issues.append({
                'prev_part': prev_part,
                'next_part': next_part,
                'combined':  combined,
                'pos_in_text': (i, len(prev_tokens))
            })

    return issues


def s4_merge_linebreak_words(text, issues):
    lines = text.splitlines()
    for issue in reversed(issues):
        i, _ = issue['pos_in_text']
        lines[i] = (
            lines[i].rstrip() +
            issue['combined'] +
            lines[i + 1].lstrip()[len(issue['next_part']):]
        )
        lines.pop(i + 1)
    return "\n".join(lines)


def s4_pythainlp_spellcheck(tokens, pos_tags):
    misspelled = []
    for i, w in enumerate(tokens):
        if (
            not w.strip()
            or w in s4_thai_dict
            or w in s4_misspell_whitelist
            or w in s4_loanwords_whitelist
            or len(w) == 1
            or 'ๆ' in w
        ):
            continue
        misspelled.append({
            'word':  w,
            'pos':   pos_tags[i][1] if i < len(pos_tags) else None,
            'index': i
        })
    return misspelled


def s4_longdo_spellcheck_batch(words):
    results = {}
    if not words:
        return results
    try:
        payload  = {"key": S4_LONGDO_API_KEY, "text": "\n".join(words)}
        response = requests.post(
            S4_LONGDO_API_URL,
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=6
        )
        if response.status_code == 200:
            for e in response.json().get("result", []):
                if e.get("suggestions"):
                    results[e["word"]] = e["suggestions"]
    except Exception as e:
        print(f"Longdo API error: {e}")
    return results


def s4_check_loanword_spelling(tokens):
    mistakes = []
    for tok in tokens:
        matches = difflib.get_close_matches(
            tok, list(s4_loanwords_whitelist), n=1, cutoff=0.7
        )
        if matches and tok not in s4_loanwords_whitelist:
            mistakes.append({'found': tok, 'should_be': matches[0]})
    return mistakes


def s4_find_unallowed_punctuations(text):
    pattern = (
        f"[^{''.join(re.escape(p) for p in s4_allowed_punctuations)}"
        r"a-zA-Z0-9ก-๙\s]"
    )
    return set(re.findall(pattern, text))


def s4_analyze_maiyamok(tokens, pos_tags):
    results             = []
    found_invalid       = False
    repeated_word_issues = []

    VALID_POS = {
        'NCMN', 'NNP', 'VACT', 'VNIR', 'CLFV',
        'ADVN', 'ADVI', 'ADVP', 'PRP', 'ADV'
    }

    for i in range(len(tokens) - 1):
        if tokens[i] == tokens[i + 1] and tokens[i] in s4_repeatable_words:
            repeated_word_issues.append(
                f"พบคำซ้ำ: {tokens[i]}{tokens[i+1]} → ควรเป็น '{tokens[i]} ๆ'"
            )
            found_invalid = True

    for i, token in enumerate(tokens):
        if token != 'ๆ':
            continue

        prev_idx  = i - 1
        prev_word = tokens[prev_idx] if prev_idx >= 0 else None
        prev_tag  = pos_tags[prev_idx][1] if prev_idx >= 0 else None

        if prev_word is None or prev_word == 'ๆ':
            verdict = "❌ ไม้ยมกไม่ควรขึ้นต้นประโยค/คำ"
        elif prev_word in s4_forbid_list:
            verdict = '❌ ไม่ควรใช้ไม้ยมกกับคำนี้'
        elif (prev_tag in VALID_POS) or (prev_word in s4_allow_list):
            verdict = '✅ ถูกต้อง (ใช้ไม้ยมกซ้ำคำได้)'
        else:
            verdict = '❌ ไม่ควรใช้ไม้ยมก นอกจากกับคำนาม/กริยา/วิเศษณ์'

        context = tokens[max(0, i - 2):min(len(tokens), i + 3)]
        results.append({
            'คำก่อนไม้ยมก': prev_word or '',
            'POS คำก่อน':  prev_tag or '',
            'บริบท':       ' '.join(context),
            'สถานะ':       verdict
        })
        if verdict.startswith('❌'):
            found_invalid = True

    return results, found_invalid, repeated_word_issues

# =========================================================
# S4 — PREDICT
# =========================================================

def predict_s4_score(text):
    """
    คืนค่า dict:
      S4_SCORE   : 0.0 / 0.5 / 1.0
      S4_REASONS : รายการข้อผิดพลาด (str)
    """

    text = s4_normalize_thai_variants(str(text))
    text, dataset_errors = fix_and_collect_typos(text, s4_misspell_dict)

    # ตรวจการฉีกคำ
    linebreak_issues = s4_analyze_linebreak_issues(text)
    corrected_text   = s4_merge_linebreak_words(text, linebreak_issues)

    tokens   = word_tokenize(
        corrected_text, engine='newmm', keep_whitespace=False
    )
    pos_tags = pos_tag(tokens, corpus='orchid')

    # ตรวจสะกด
    pythai_errors  = s4_pythainlp_spellcheck(tokens, pos_tags)
    wrong_words    = [e['word'] for e in pythai_errors]
    longdo_results = s4_longdo_spellcheck_batch(wrong_words)

    spelling_errors = [
        {**e, 'suggestions': longdo_results[e['word']]}
        for e in pythai_errors
        if e['word'] in longdo_results
    ]

    loanword_errors = s4_check_loanword_spelling(tokens)

    # ตรวจเครื่องหมาย
    punct_errors = s4_find_unallowed_punctuations(text)

    # ตรวจไม้ยมก
    maiyamok_results, _, repeated_word_issues = s4_analyze_maiyamok(
        tokens, pos_tags
    )

    # นับ unique spelling errors
    unique_spelling = set()
    for e in spelling_errors:
        unique_spelling.add(e['word'])
    for e in dataset_errors:
        unique_spelling.add(e['wrong'])
    for e in loanword_errors:
        unique_spelling.add(e['found'])

    error_counts = {
        "spelling":  len(unique_spelling),
        "linebreak": len(linebreak_issues),
        "punct":     len(punct_errors),
        "maiyamok":  (
            sum(1 for r in maiyamok_results if r['สถานะ'].startswith('❌'))
            + len(repeated_word_issues)
        )
    }

    n_issue_types        = sum(1 for c in error_counts.values() if c > 0)
    multi_in_single_type = any(c >= 2 for c in error_counts.values())

    # สร้าง reasons
    reasons = []

    if error_counts["linebreak"]:
        details = [
            f"{i['prev_part']} + {i['next_part']} → {i['combined']}"
            for i in linebreak_issues
        ]
        reasons.append("พบการฉีกคำข้ามบรรทัด: " + "; ".join(details))

    if error_counts["spelling"]:
        error_words   = list(set([e['word'] for e in spelling_errors]))
        dataset_desc  = list(set([
            f"{e['wrong']} (ควรเป็น {e['right']})" for e in dataset_errors
        ]))
        loanword_desc = list(set([
            f"{e['found']} (ควรเป็น {e['should_be']})" for e in loanword_errors
        ]))
        reasons.append(
            "ตรวจเจอคำสะกดผิด: " +
            ', '.join(error_words + dataset_desc + loanword_desc)
        )

    if error_counts["punct"]:
        reasons.append(
            f"ใช้เครื่องหมายที่ไม่อนุญาต: {', '.join(punct_errors)}"
        )

    if error_counts["maiyamok"]:
        all_maiyamok_errors = []
        for x in maiyamok_results:
            if x['สถานะ'].startswith('❌'):
                all_maiyamok_errors.append(
                    f"{x['คำก่อนไม้ยมก']}: {x['สถานะ']}"
                )
        all_maiyamok_errors.extend(repeated_word_issues)
        reasons.append("ใช้ไม้ยมกผิด: " + '; '.join(all_maiyamok_errors))

    if not reasons:
        reasons.append("ไม่มีปัญหา")

    # เกณฑ์คะแนน
    total_errors = sum(error_counts.values())
    if total_errors == 0:
        score = 1.0
    elif n_issue_types == 1 and not multi_in_single_type:
        score = 0.5
    else:
        score = 0.0

    return {
        "S4_SCORE":   score,
        "S4_REASONS": " | ".join(reasons)
    }

# =========================================================
# LOAD S5 MODEL (โหลดครั้งเดียว)
# =========================================================

print("Loading S5 Scoring Model...")

s5_tokenizer = AutoTokenizer.from_pretrained(S5_MODEL_PATH)

s5_model = AutoModelForSequenceClassification.from_pretrained(
    S5_MODEL_PATH
)

s5_model.to(device)
s5_model.eval()

s5_id2score = {
    0: 0.0,
    1: 0.5,
    2: 1.0
}

# =========================================================
# LOAD S6 MODEL (โหลดครั้งเดียว)
# =========================================================

print("Loading S6 Scoring Model...")

s6_tokenizer = AutoTokenizer.from_pretrained(S6_MODEL_PATH)

s6_model = AutoModelForSequenceClassification.from_pretrained(
    S6_MODEL_PATH
)

s6_model.to(device)
s6_model.eval()

s6_id2score = {
    0: 0.0,
    1: 0.5,
    2: 1.0
}

# =========================================================
# S5 — PREDICT
# =========================================================

def predict_s5_score(text):
    """
    คืนค่า dict:
      S5_SCORE     : 0.0 / 0.5 / 1.0
      S5_PRED_CLASS: class id (0/1/2)

    ใช้ tokenizer + model โดยตรง (ไม่ใช้ datasets/Trainer)
    เพื่อหลีกเลี่ยง torchvision.VideoReader ImportError
    """

    inputs = s5_tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=S5_MAX_LENGTH
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs    = s5_model(**inputs)
        logits     = outputs.logits
        pred_class = int(torch.argmax(logits, dim=-1).item())

    return {
        "S5_SCORE":      s5_id2score[pred_class],
        "S5_PRED_CLASS": pred_class
    }

# =========================================================
# S6 — PREDICT
# =========================================================

def predict_s6_score(text):
    """
    คืนค่า dict:
      S6_SCORE     : 0.0 / 0.5 / 1.0
      S6_PRED_CLASS: class id (0/1/2)
    """

    inputs = s6_tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=S6_MAX_LENGTH
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs    = s6_model(**inputs)
        logits     = outputs.logits
        pred_class = int(torch.argmax(logits, dim=-1).item())

    return {
        "S6_SCORE":      s6_id2score[pred_class],
        "S6_PRED_CLASS": pred_class
    }

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):
    text = str(text)
    text = text.replace("\u0e4d\u0e32", "\u0e33")
    text = unicodedata.normalize("NFC", text)
    return text.strip()

# =========================================================
# WORD TOKENIZE
# =========================================================

def preprocess_word_tokenize(text):
    words = word_tokenize(text, engine="newmm")
    return " ".join(words)

# =========================================================
# SEGMENT ONE SENTENCE
# =========================================================

def segment_one_sentence(text):
    original_text = clean_text(text)
    proc_text     = preprocess_word_tokenize(original_text)

    encoded = seg_tokenizer(
        proc_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH_SEG,
        return_offsets_mapping=True
    )

    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = seg_model(**encoded)

    preds = torch.argmax(
        outputs.logits, dim=-1
    )[0].cpu().tolist()

    split_points = []

    for pred, (start, end) in zip(preds, offsets):
        if start == 0 and end == 0:
            continue
        if pred == B_LABEL_ID:
            split_points.append(start)

    split_points = sorted(set(split_points))

    if not split_points:
        return [original_text]

    if 0 not in split_points:
        split_points.insert(0, 0)

    segments = []

    for i in range(len(split_points)):
        start = split_points[i]
        end   = (
            split_points[i + 1]
            if i + 1 < len(split_points)
            else len(proc_text)
        )
        seg = proc_text[start:end].strip()
        if seg:
            seg = seg.replace(" ", "")
            segments.append(seg)

    return segments

# =========================================================
# SEGMENT FULL TEXT
# =========================================================

def segment_text(text):
    text      = clean_text(text)
    sentences = sent_tokenize(text, engine="crfcut")
    all_segs  = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        all_segs.extend(segment_one_sentence(sent))

    return all_segs

# =========================================================
# PREPROCESS FOR MAIN IDEA
# =========================================================

def preprocess_for_main_idea(text):
    segments         = segment_text(text)
    cleaned_segments = [
        seg.strip()
        for seg in segments
        if len(seg.strip()) >= 10
    ]
    return " ".join(cleaned_segments)

# =========================================================
# S2 SCORE PREDICTION
# =========================================================

def predict_s2_score(processed_text):
    inputs = s2_tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = s2_model(**inputs)
        logits  = outputs.logits
        pred    = torch.argmax(logits, dim=-1).item()
        probs   = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    return {
        "S2_SCORE":  id2score[pred],
        "S2_PROB_0": float(probs[0]),
        "S2_PROB_1": float(probs[1]),
        "S2_PROB_2": float(probs[2])
    }

# =========================================================
# CHECK COPY (การยกข้อความจากบทอ่าน)
# =========================================================

def extract_key_sentences(text, top_k=3):
    """ดึงประโยคที่ใกล้เคียงใจความหลักของข้อความมากที่สุด"""
    sentences  = text.split(" ")
    embeddings = embed_model.encode(sentences, convert_to_tensor=True)
    doc_emb    = embeddings.mean(dim=0)
    scores     = util.cos_sim(embeddings, doc_emb)
    top_idx    = scores.argsort(descending=True)[:top_k]
    return [sentences[i] for i in top_idx]


def check_copy(reference, student):
    """
    เปรียบเทียบ student กับ reference
    similarity > COPY_THRESHOLD → มีการยกข้อความ
    คืนค่า: (label, similarity_score)
    """
    ref_vec = embed_model.encode(reference, convert_to_tensor=True)
    stu_vec = embed_model.encode(student,   convert_to_tensor=True)
    score   = float(util.cos_sim(ref_vec, stu_vec).item())

    if score > COPY_THRESHOLD:
        return "มีการยกข้อความจากบทอ่าน", score
    else:
        return "ไม่มีการยกข้อความ", score

# =========================================================
# CHECK SINGLE SENTENCE (ประโยคความเดียว)
# =========================================================

CONNECTORS = [
    "และ", "หรือ", "แต่", "เพราะ", "จึง",
    "ที่", "ซึ่ง", "เมื่อ", "ถ้า", "แม้", "โดย"
]


def _brute_force_split(word):
    for i in range(1, len(word)):
        left, right  = word[:i], word[i:]
        sub_pos      = pos_tag([left, right], corpus="orchid")
        if (
            len(sub_pos) == 2
            and sub_pos[0][1] in ["VACT", "VSTA"]
            and sub_pos[1][1] == "NCMN"
        ):
            return sub_pos
    return [(word, None)]


def _find_main_verbs(text):
    tokens      = word_tokenize(text)
    pos         = pos_tag(tokens, corpus="orchid")
    final_tokens = []

    for word, tag in pos:
        if tag is None or tag == "NCMN":
            final_tokens.extend(_brute_force_split(word))
        else:
            final_tokens.append((word, tag))

    verbs = [w for w, t in final_tokens if t in ["VACT", "VSTA"]]
    return final_tokens, verbs


def check_single_sentence(text):
    """
    ตรวจว่าข้อความเป็นประโยคความเดียว 1 ประโยคหรือไม่
    คืนค่า: "เป็นประโยคความเดียว" หรือ "ไม่ใช่ประโยคความเดียว (...)"
    """
    sentences = re.split(r"[.!?]|\n", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) != 1:
        return "ไม่ใช่ประโยคความเดียว (มีมากกว่า 1 ประโยค)"

    sentence = sentences[0]

    for word in CONNECTORS:
        if word in sentence:
            return f"ไม่ใช่ประโยคความเดียว (มีคำเชื่อม: {word})"

    _, verbs = _find_main_verbs(sentence)

    if len(verbs) == 1:
        return "เป็นประโยคความเดียว"
    else:
        return f"ไม่ใช่ประโยคความเดียว (มีหลายกริยา: {', '.join(verbs)})"

# =========================================================
# CHECK ENUMERATION (ลำดับข้อ)
# =========================================================

def has_enumeration(text):
    """
    ตรวจว่าข้อความมีการใช้ลำดับข้อหรือไม่
    รูปแบบที่ตรวจ: 1. / (1) / ก. / - / --
    พบ >= 2 ตัว → ถือว่ามีลำดับข้อ
    """
    pattern = r"(?:(?<=\n)|(?<=\s)|^)(\d+\.|\(\d+\)|[ก-ฮ]\.|-+)(?=\S)"
    matches = re.findall(pattern, text)
    return "มีการใช้ลำดับข้อ" if len(matches) >= 2 else "ไม่มีการใช้ลำดับข้อ"

# =========================================================
# MAIN IDEA + S2 + S3 + S4 + S5 + S6
# =========================================================

def predict_main_ideas(text):

    processed_text = preprocess_for_main_idea(text)

    # =========================================================
    # ข้อตกลงการตรวจ: ตรวจว่าย่อความไม่ผิดเนื้อหาจากบทอ่าน
    # ใช้ cosine similarity เดียวกับ S3
    # =========================================================

    agreement_error, agreement_similarity = check_paraphrase(
        text, s3_reference_text
    )

    # ถ้า similarity ต่ำกว่า threshold → ย่อผิดเนื้อหา → คะแนน 0 ทั้งหมด
    if agreement_error:
        zero_result = {
            "SEGMENT_TEXT":         processed_text,
            "AGREEMENT_PASS":       False,
            "AGREEMENT_SIMILARITY": agreement_similarity,
            "AGREEMENT_REASON":     (
                f"ย่อความผิดเนื้อหาจากบทอ่าน "
                f"(similarity={agreement_similarity} < {PARAPHRASE_THRESHOLD})"
            ),
            "ENUMERATION_CHECK":      "ไม่ได้ตรวจ (ไม่ผ่านข้อตกลง similarity)",
            "SINGLE_SENTENCE_CHECK":  "ไม่ได้ตรวจ (ไม่ผ่านข้อตกลง similarity)",
            "COPY_CHECK":             "ไม่ได้ตรวจ (ไม่ผ่านข้อตกลง similarity)",
            "COPY_SIMILARITY":        0.0
        }
        for label in LABEL_COLUMNS:
            zero_result[label]           = 0
            zero_result[f"{label}_PROB"] = 0.0

        zero_result.update({
            "S1_SCORE":    0,
            "S2_SCORE":    0, "S2_PROB_0": 0.0, "S2_PROB_1": 0.0, "S2_PROB_2": 0.0,
            "S3_SCORE":    0, "S3_SIMILARITY": agreement_similarity,
            "S3_REASON":   "ไม่ผ่านข้อตกลงการตรวจ",
            "S4_SCORE":    0.0, "S4_REASONS": "ไม่ผ่านข้อตกลงการตรวจ",
            "S5_SCORE":    0.0, "S5_PRED_CLASS": -1,
            "S6_SCORE":    0.0, "S6_PRED_CLASS": -1,
            "TOTAL_SCORE": 0
        })
        return zero_result

    # ผ่านข้อตกลง → ดำเนินการตรวจปกติ
    # ---------- S1 ----------
    inputs = main_tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH_MAIN
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs   = main_model(**inputs)
        logits    = outputs.logits
        probs     = torch.sigmoid(logits).cpu().numpy()[0]
        THRESHOLD = 0.89
        preds     = (probs >= THRESHOLD).astype(int)

    result = {
        "SEGMENT_TEXT":         processed_text,
        "AGREEMENT_PASS":       True,
        "AGREEMENT_SIMILARITY": agreement_similarity,
        "AGREEMENT_REASON":     "ผ่านข้อตกลงการตรวจ"
    }

    for i, label in enumerate(LABEL_COLUMNS):
        result[label]              = int(preds[i])
        result[f"{label}_PROB"]    = float(probs[i])

    # =========================================================
    # ข้อตกลงการตรวจ: ถ้ามีลำดับข้อ → ตรวจแค่ S1, S2-S6 = 0
    # =========================================================

    enumeration_found = has_enumeration(text)
    result["ENUMERATION_CHECK"]     = enumeration_found
    result["SINGLE_SENTENCE_CHECK"] = "ไม่ได้ตรวจ (มีลำดับข้อ)" if enumeration_found == "มีการใช้ลำดับข้อ" else ""

    s1_score = sum(result[label] for label in LABEL_COLUMNS)
    result["S1_SCORE"] = s1_score

    if enumeration_found == "มีการใช้ลำดับข้อ":
        result.update({
            "S2_SCORE":    0, "S2_PROB_0": 0.0, "S2_PROB_1": 0.0, "S2_PROB_2": 0.0,
            "S3_SCORE":    0, "S3_SIMILARITY": 0.0,
            "S3_REASON":   "ไม่ผ่านข้อตกลงการตรวจ (มีการใช้ลำดับข้อ)",
            "S4_SCORE":    0.0, "S4_REASONS": "ไม่ผ่านข้อตกลงการตรวจ (มีการใช้ลำดับข้อ)",
            "S5_SCORE":    0.0, "S5_PRED_CLASS": -1,
            "S6_SCORE":    0.0, "S6_PRED_CLASS": -1,
            "COPY_CHECK":  "ไม่ได้ตรวจ (มีลำดับข้อ)",
            "COPY_SIMILARITY": 0.0,
            "TOTAL_SCORE": s1_score
        })
        return result

    # =========================================================
    # ข้อตกลงการตรวจ: ถ้ามีการยกข้อความ → ตรวจแค่ S1, S2-S6 = 0
    # =========================================================

    copy_label, copy_similarity = check_copy(copy_reference_text, text)
    result["COPY_CHECK"]       = copy_label
    result["COPY_SIMILARITY"]  = copy_similarity

    if copy_label == "มีการยกข้อความจากบทอ่าน":
        result.update({
            "S2_SCORE":    0, "S2_PROB_0": 0.0, "S2_PROB_1": 0.0, "S2_PROB_2": 0.0,
            "S3_SCORE":    0, "S3_SIMILARITY": copy_similarity,
            "S3_REASON":   "ไม่ผ่านข้อตกลงการตรวจ (มีการยกข้อความจากบทอ่าน)",
            "S4_SCORE":    0.0, "S4_REASONS": "ไม่ผ่านข้อตกลงการตรวจ (มีการยกข้อความจากบทอ่าน)",
            "S5_SCORE":    0.0, "S5_PRED_CLASS": -1,
            "S6_SCORE":    0.0, "S6_PRED_CLASS": -1,
            "SINGLE_SENTENCE_CHECK": "ไม่ได้ตรวจ (มีการยกข้อความ)",
            "TOTAL_SCORE": s1_score
        })
        return result

    # =========================================================
    # ข้อตกลงการตรวจ: ถ้าเป็นประโยคความเดียว → ตรวจแค่ S1, S2-S6 = 0
    # =========================================================

    single_sentence_check = check_single_sentence(text)
    result["SINGLE_SENTENCE_CHECK"] = single_sentence_check

    if single_sentence_check == "เป็นประโยคความเดียว":
        result.update({
            "S2_SCORE":    0, "S2_PROB_0": 0.0, "S2_PROB_1": 0.0, "S2_PROB_2": 0.0,
            "S3_SCORE":    0, "S3_SIMILARITY": 0.0,
            "S3_REASON":   "ไม่ผ่านข้อตกลงการตรวจ (เป็นประโยคความเดียว)",
            "S4_SCORE":    0.0, "S4_REASONS": "ไม่ผ่านข้อตกลงการตรวจ (เป็นประโยคความเดียว)",
            "S5_SCORE":    0.0, "S5_PRED_CLASS": -1,
            "S6_SCORE":    0.0, "S6_PRED_CLASS": -1,
            "TOTAL_SCORE": s1_score
        })
        return result

    # =========================================================
    # ข้อตกลงการตรวจ: ถ้า S1 = 0 → S2-S6 ได้ 0 คะแนนทั้งหมด
    # =========================================================

    if s1_score == 0:
        result.update({
            "S2_SCORE":    0, "S2_PROB_0": 0.0, "S2_PROB_1": 0.0, "S2_PROB_2": 0.0,
            "S3_SCORE":    0, "S3_SIMILARITY": 0.0,
            "S3_REASON":   "ไม่ผ่านข้อตกลงการตรวจ (S1 = 0)",
            "S4_SCORE":    0.0, "S4_REASONS": "ไม่ผ่านข้อตกลงการตรวจ (S1 = 0)",
            "S5_SCORE":    0.0, "S5_PRED_CLASS": -1,
            "S6_SCORE":    0.0, "S6_PRED_CLASS": -1,
            "TOTAL_SCORE": 0
        })
        return result

    # S1 > 0 → ดำเนินการตรวจ S2-S6 ตามปกติ

    # ---------- S2 ----------
    s2_result = predict_s2_score(processed_text)
    result.update(s2_result)

    # ---------- S3 ----------
    s3_result = predict_s3_score(text)   # ← ใช้ข้อความดิบ (ไม่ segment)
    result.update(s3_result)

    # ---------- S4 ----------
    s4_result = predict_s4_score(text)   # ← ใช้ข้อความดิบ
    result.update(s4_result)

    # ---------- S5 ----------
    s5_result = predict_s5_score(text)
    result.update(s5_result)

    # ---------- S6 ----------
    s6_result = predict_s6_score(text)
    result.update(s6_result)

    # ---------- TOTAL ----------
    result["TOTAL_SCORE"] = (
        result["S1_SCORE"] +
        result["S2_SCORE"] +
        result["S3_SCORE"] +
        result["S4_SCORE"] +
        result["S5_SCORE"] +
        result["S6_SCORE"]
    )

    return result



#========================= ข้อ 30.2 =========================


# =========================================================
# MODEL PATHS
# =========================================================

MODEL_PATH_S8 = r"C:\Users\User\Documents\New_project\all_dataset\S8\model_train_S8\train"
MODEL_PATH_S9 = r"C:\Users\User\Documents\New_project\all_dataset\S9\model_train_S9\S9_scoring_model_weight"
MODEL_PATH_S12 = r"C:\Users\User\Documents\New_project\all_dataset\S12\model_train_S12\model"
MODEL_PATH_S13 = r"C:\Users\User\Documents\New_project\all_dataset\S13\model_train_S13\model"

# =========================================================
# LOAD S8 MODEL
# =========================================================

tokenizer_s8 = AutoTokenizer.from_pretrained(MODEL_PATH_S8, use_fast=False)
model_s8     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH_S8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_s8.to(device)
model_s8.eval()

# =========================================================
# LOAD S9 MODEL
# =========================================================

tokenizer_s9 = AutoTokenizer.from_pretrained(MODEL_PATH_S9)
model_s9     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH_S9)
model_s9.to(device)
model_s9.eval()

# =========================================================
# LOAD S12 MODEL
# =========================================================

tokenizer_s12 = AutoTokenizer.from_pretrained(MODEL_PATH_S12)
model_s12     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH_S12)
model_s12.to(device)
model_s12.eval()

# =========================================================
# LOAD S13 MODEL
# =========================================================

tokenizer_s13 = AutoTokenizer.from_pretrained(MODEL_PATH_S13)
model_s13     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH_S13)

model_s13.to(device)
model_s13.eval()

# =========================================================
# COPY DETECTION
# =========================================================

copy_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

copy_df = pd.read_csv(
    r"C:\Users\User\Documents\New_project\all_dataset\S10\s10_R2_last.csv"
)

reference_texts = (
    copy_df["reference_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

reference_embeddings = copy_model.encode(
    reference_texts,
    convert_to_numpy=True,
    normalize_embeddings=True
)

def check_copying(
    student_text,
    threshold=0.90
):

    student_embedding = copy_model.encode(
        str(student_text),
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        [student_embedding],
        reference_embeddings
    )[0]

    best_idx = np.argmax(scores)
    best_score = float(scores[best_idx])

    return {
        "copy_similarity": round(best_score, 4),
        "is_copy": best_score >= threshold,
        "copy_result": (
            "คัดลอกบทอ่าน"
            if best_score >= threshold
            else "ไม่ถือว่าคัดลอก"
        ),
        "matched_reference":
            reference_texts[best_idx]
    }


# =========================================================
# LABEL MAP (S8)
# =========================================================

reverse_label_map_s8 = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}

# =========================================================
# LABEL MAP (S12)
# =========================================================

id2label_s12 = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0}

# =========================================================
# LABEL MAP (S13)
# =========================================================

id2label_s13 = {
    0: 0.0,
    1: 0.5,
    2: 1.0,
    3: 1.5,
    4: 2.0
}

# =========================================================
# S10 — API KEYS
# =========================================================

TNER_API_KEY       = "zyHC3BNtLiesIuTj2UMlQd8DhrVXBxzM"
CYBERBULLY_API_KEY = "zyHC3BNtLiesIuTj2UMlQd8DhrVXBxzM"

# =========================================================
# S11 — LOAD FILES
# =========================================================

with open(r"C:\Users\User\Documents\New_project\all_dataset\S11\update_common_misspellings.json", 'r', encoding='utf-8') as f:
    misspell_data     = json.load(f)
    misspel_whitelist = set(item['right'] for item in misspell_data if item['right'])

misspell_dict = {
    item['wrong']: item['right']
    for item in misspell_data if item['right']
}

with open(r"C:\Users\User\Documents\New_project\all_dataset\S11\thai_loanwords_new_update.json", 'r', encoding='utf-8') as f:
    loanwords_data     = json.load(f)
    loanwords_whitelist = set(item['thai_word'] for item in loanwords_data)

keywords_list_df   = pd.read_csv(r"C:\Users\User\Documents\New_project\all_dataset\S11\keyword_list_S4_S11_new.csv")
splitable_phrases  = keywords_list_df['splitable_phrases'].dropna().tolist()
strict_not_split_words = keywords_list_df['strict_not_split_word'].dropna().tolist()
allow_list         = keywords_list_df['allow_list'].dropna().tolist()
forbid_list        = keywords_list_df['forbid_list'].dropna().tolist()
repeatable_words   = keywords_list_df['repeatable_words'].dropna().tolist()

LONGDO_API_KEY = '33586c7cf5bfa0029887a9831bf94963'
LONGDO_API_URL = 'https://api.longdo.com/spell-checker/proof'

thai_dict            = set(w for w in set(thai_words()) if (' ' not in w) and w.strip())
allowed_punctuations = {'.', ',', '-', '(', ')', '!', '?', '%', '"', '"', '\u2018', '\u2019', '"', "'", '\u2026', '\u0e2f'}

# =========================================================
# ข้อตกลงการตรวจ
# =========================================================

def check_numline(text):
    num_line = len(text.strip().split('\n'))
    if 1 <= num_line <= 2:
        return num_line, "คำตอบ 1-2 บรรทัด"
    return num_line, "ไม่ใช่คำตอบสั้น"

# =========================================================
# S7 — Keyword Detection
# =========================================================

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"ไม่\s+เห็นด้วย", "ไม่เห็นด้วย", text)
    text = re.sub(r"เห็น\s+ด้วย", "เห็นด้วย", text)
    return text

def s7_keyword_classify(text):
    text = normalize_text(text)
    if "เห็นด้วยและไม่เห็นด้วย" in text:
        return True, "พบคำ: เห็นด้วยและไม่เห็นด้วย"
    if "ไม่เห็นด้วย" in text:
        return True, "พบคำ: ไม่เห็นด้วย"
    if "เห็นด้วย" in text:
        return True, "พบคำ: เห็นด้วย"
    return False, "ไม่พบ keyword"

# =========================================================
# S8 — BERT Scoring (WangchanBERTa)
# =========================================================

MAX_LENGTH_S8 = 256

def s8_predict_score(text):
    inputs = tokenizer_s8(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH_S8
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs   = model_s8(**inputs)
        pred_class = torch.argmax(outputs.logits, dim=1).item()
    return reverse_label_map_s8[pred_class]

# =========================================================
# S9 — BERT Scoring (S2 model)
# =========================================================

MAX_LENGTH_S9 = 512

def s9_predict_score(text):
    inputs = tokenizer_s9(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH_S9
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs    = model_s9(**inputs)
        pred_class = torch.argmax(outputs.logits, dim=1).item()
    return pred_class

def s9_predict_batch(texts):
    df_tmp  = pd.DataFrame({"text": [str(t) for t in texts]})
    dataset = Dataset.from_pandas(df_tmp)

    def tokenize_fn(example):
        return tokenizer_s9(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH_S9
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer_s9)
    trainer       = Trainer(model=model_s9, data_collator=data_collator)
    predictions   = trainer.predict(dataset)
    return np.argmax(predictions.predictions, axis=-1).tolist()

# =========================================================
# S10 — helpers
# =========================================================

personal_pronoun_1   = {"หนู", "ข้า", "กู"}
personal_pronoun_2   = {"คุณ", "เธอ", "แก", "ตัวเอง", "เอ็ง", "มึง", "เขา"}
all_personal_pronouns = personal_pronoun_1.union(personal_pronoun_2)

def _remove_opinion_prefix(text):
    for pattern in [
        r"^\s*เห็นด้วย\s*(เพราะ)?\s*",
        r"^\s*ไม่เห็นด้วย\s*(เพราะ)?\s*"
    ]:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()

def _check_sentiment(text):
    text = _remove_opinion_prefix(text)
    try:
        response = requests.get(
            "https://api.aiforthai.in.th/ssense",
            headers={"Apikey": TNER_API_KEY},
            params={"text": text},
            timeout=30
        )
        if response.status_code == 200:
            result    = response.json()
            sentiment = result.get("sentiment")
            if isinstance(sentiment, dict):
                score    = sentiment.get("score", "")
                polarity = sentiment.get("polarity", "")
                if str(score) == "0" or polarity == "":
                    return "neutral"
                return polarity
            if isinstance(sentiment, str):
                return sentiment
            if "polarity" in result:
                return result["polarity"]
    except Exception as e:
        print("SSENSE ERROR:", e)
    return "unknown"

def _check_named_entities(text):
    try:
        response = requests.post(
            "https://api.aiforthai.in.th/tner",
            headers={"Apikey": TNER_API_KEY},
            data={"text": text}
        )
        if response.status_code == 200:
            ner_result   = response.json()
            words        = ner_result.get("words", [])
            tags         = ner_result.get("tags", [])
            target_types = {
                "PER", "ORG", "LOC", "TTL", "DES",
                "ABB_PER", "ABB_ORG", "ABB_LOC", "ABB_TTL", "ABB_DES"
            }
            entities, current_entity, current_type = [], "", None
            for word, tag in zip(words, tags):
                if tag.startswith("B-"):
                    if current_entity:
                        entities.append(current_entity)
                    entity_type    = tag[2:]
                    current_entity = word if entity_type in target_types else ""
                    current_type   = entity_type if entity_type in target_types else None
                elif tag.startswith("I-") and current_entity and tag[2:] == current_type:
                    current_entity += word
                else:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity, current_type = "", None
            for i in range(len(words) - 1):
                if words[i] in {"นาย", "นาง", "นางสาว"}:
                    entities.append(words[i] + words[i + 1])
            if current_entity:
                entities.append(current_entity)
            entities = list(dict.fromkeys(entities))
            filtered = [e for e in entities if not any(e != o and o.startswith(e) for o in entities)]
            if filtered:
                return True, filtered
    except Exception as e:
        print("TNER ERROR:", e)
    return False, []

def _check_cyberbully(text):
    try:
        response = requests.get(
            "https://api.aiforthai.in.th/bully",
            headers={"Apikey": CYBERBULLY_API_KEY},
            params={"text": text}
        )
        if response.status_code == 200:
            result      = response.json()
            bully_words = result.get("bully_word")
            if bully_words in (None, "None", ["None"], []):
                return False, []
            return True, bully_words
    except Exception as e:
        print("CYBERBULLY ERROR:", e)
    return False, []

def _check_personal_pronouns(text):
    tokens = word_tokenize(text, engine="newmm")
    found  = [t for t in tokens if t in all_personal_pronouns]
    return (True, found) if found else (False, [])

# =========================================================
# S10 — Evaluate
# =========================================================

def s10_evaluate(text):
    mistakes  = []
    sentiment = _check_sentiment(text)

    if sentiment == "negative":
        ne_flag, ne_words = _check_named_entities(text)
        if ne_flag:
            mistakes.append({"type": "Named Entity", "words": ne_words})

    bully_flag, bully_words = _check_cyberbully(text)
    if bully_flag:
        mistakes.append({"type": "Cyberbully", "words": bully_words})

    pronoun_flag, pronouns = _check_personal_pronouns(text)
    if pronoun_flag:
        mistakes.append({"type": "Personal Pronoun", "words": pronouns})

    mistake_count = len(mistakes)
    s10_score     = 2 if mistake_count == 0 else (1 if mistake_count == 1 else 0)

    sentiment_th = {
        "positive": "ข้อความเชิงบวก",
        "negative": "ข้อความเชิงลบ",
        "neutral":  "ข้อความเป็นกลาง"
    }.get(sentiment, "ไม่สามารถวิเคราะห์ได้")

    return {
        "s10_score":    s10_score,
        "sentiment":    sentiment,
        "sentiment_th": sentiment_th,
        "s10_mistakes": mistakes if mistakes else "ไม่มีข้อผิดพลาด"
    }

# =========================================================
# S11 — helpers
# =========================================================

def _check_linebreak_issue(prev_line_tokens, next_line_tokens, max_words=3):
    last_word  = prev_line_tokens[-1]
    first_word = next_line_tokens[0]
    if last_word.endswith('-') or first_word.startswith('-'):
        return False, None, None, None
    for prev_n in range(1, min(max_words, len(prev_line_tokens)) + 1):
        prev_part = ''.join(prev_line_tokens[-prev_n:])
        for next_n in range(1, min(max_words, len(next_line_tokens)) + 1):
            next_part = ''.join(next_line_tokens[:next_n])
            combined  = normalize(prev_part + next_part)
            if (
                (' ' not in combined)
                and (combined not in splitable_phrases)
                and (
                    (combined in strict_not_split_words) or (
                        (combined in thai_dict)
                        and (len(word_tokenize(combined, engine='newmm')) == 1)
                    )
                )
            ):
                return True, prev_part, next_part, combined
    return False, None, None, None

def _analyze_linebreak_issues(text):
    lines  = text.strip().splitlines()
    issues = []
    for i in range(len(lines) - 1):
        prev_tokens = word_tokenize(lines[i].strip())
        next_tokens = word_tokenize(lines[i + 1].strip())
        if not prev_tokens or not next_tokens:
            continue
        issue, prev_part, next_part, combined = _check_linebreak_issue(prev_tokens, next_tokens)
        if issue:
            issues.append({
                'prev_part': prev_part,
                'next_part': next_part,
                'combined':  combined,
                'pos_in_text': (i, len(prev_tokens))
            })
    return issues

def _merge_linebreak_words(text, linebreak_issues):
    lines = text.splitlines()
    for issue in reversed(linebreak_issues):
        i, _ = issue['pos_in_text']
        lines[i] = (
            lines[i].rstrip()
            + issue['combined']
            + lines[i + 1].lstrip()[len(issue['next_part']):]
        )
        lines.pop(i + 1)
    return "\n".join(lines)

def _pythainlp_spellcheck(tokens, pos_tags):
    misspelled = []
    for i, w in enumerate(tokens):
        if not w.strip() or w in thai_dict or len(w) == 1 or 'ๆ' in w:
            continue
        misspelled.append({
            'word':  w,
            'pos':   pos_tags[i][1] if i < len(pos_tags) else None,
            'index': i
        })
    return misspelled

def _longdo_spellcheck_batch(words):
    results = {}
    if not words:
        return results
    try:
        response = requests.post(
            LONGDO_API_URL,
            headers={'Content-Type': 'application/json'},
            json={"key": LONGDO_API_KEY, "text": "\n".join(words)},
            timeout=6
        )
        if response.status_code == 200:
            for e in response.json().get("result", []):
                if e.get("suggestions"):
                    results[e["word"]] = e["suggestions"]
    except Exception as e:
        print(f"LONGDO ERROR: {e}")
    return results

def _fix_and_collect_typos(text):
    found, used_spans = [], []
    for wrong, right in sorted(misspell_dict.items(), key=lambda x: -len(x[0])):
        for match in re.finditer(re.escape(wrong), text):
            span = match.span()
            if any(not (span[1] <= s[0] or span[0] >= s[1]) for s in used_spans):
                continue
            found.append({"wrong": wrong, "right": right})
            used_spans.append(span)
    for e in found:
        text = text.replace(e["wrong"], e["right"])
    return text, found

def _check_loanword_spelling(tokens):
    mistakes = []
    for tok in tokens:
        matches = difflib.get_close_matches(tok, list(loanwords_whitelist), n=1, cutoff=0.7)
        if matches and tok not in loanwords_whitelist:
            mistakes.append({'found': tok, 'should_be': matches[0]})
    return mistakes

def _find_unallowed_punctuations(text):
    pattern = f"[^{''.join(re.escape(p) for p in allowed_punctuations)}a-zA-Z0-9ก-๙\\s]"
    return set(re.findall(pattern, text))

def _analyze_maiyamok(tokens, pos_tags):
    VALID_POS = {'NCMN', 'NNP', 'VACT', 'VNIR', 'CLFV', 'ADVN', 'ADVI', 'ADVP', 'PRP', 'ADV'}
    results, found_invalid, repeated_word_issues = [], False, []

    for i in range(len(tokens) - 1):
        if tokens[i] == tokens[i + 1] and tokens[i] in repeatable_words:
            repeated_word_issues.append(
                f"พบคำซ้ำ: {tokens[i]}{tokens[i+1]} → ควรเป็น '{tokens[i]} ๆ'"
            )
            found_invalid = True

    for i, token in enumerate(tokens):
        if token != 'ๆ':
            continue
        prev_idx  = i - 1
        prev_word = tokens[prev_idx] if prev_idx >= 0 else None
        prev_tag  = pos_tags[prev_idx][1] if prev_idx >= 0 else None
        if prev_word is None or prev_word == 'ๆ':
            verdict = "❌ ไม้ยมกไม่ควรขึ้นต้นประโยค/คำ"
        elif prev_word in forbid_list:
            verdict = '❌ ไม่ควรใช้ไม้ยมกกับคำนี้'
        elif (prev_tag in VALID_POS) or (prev_word in allow_list):
            verdict = '✅ ถูกต้อง (ใช้ไม้ยมกซ้ำคำได้)'
        else:
            verdict = '❌ ไม่ควรใช้ไม้ยมก นอกจากกับคำนาม/กริยา/วิเศษณ์'
        context = tokens[max(0, i - 2):min(len(tokens), i + 3)]
        results.append({
            'คำก่อนไม้ยมก': prev_word or '',
            'POS คำก่อน':    prev_tag or '',
            'บริบท':         ' '.join(context),
            'สถานะ':         verdict
        })
        if verdict.startswith('❌'):
            found_invalid = True

    return results, found_invalid, repeated_word_issues

def _evaluate_text_s11(text):
    text, dataset_errors = _fix_and_collect_typos(text)

    linebreak_issues = _analyze_linebreak_issues(text)
    corrected_text   = _merge_linebreak_words(text, linebreak_issues)
    tokens           = word_tokenize(corrected_text, engine='newmm', keep_whitespace=False)
    pos_tags         = pos_tag(tokens, corpus='orchid')

    loanword_spell_errors  = _check_loanword_spelling(tokens)
    pythai_errors          = _pythainlp_spellcheck(tokens, pos_tags)
    longdo_results         = _longdo_spellcheck_batch([e['word'] for e in pythai_errors])
    spelling_errors_legit  = [
        {**e, 'suggestions': longdo_results.get(e['word'], [])}
        for e in pythai_errors if e['word'] in longdo_results
    ]

    punct_errors    = _find_unallowed_punctuations(text)
    maiyamok_results, found_invalid, repeated_word_issues = _analyze_maiyamok(tokens, pos_tags)

    unique_spelling_errors = set()
    for e in spelling_errors_legit:
        unique_spelling_errors.add(e['word'])
    for e in dataset_errors:
        unique_spelling_errors.add(e['wrong'])
    for e in loanword_spell_errors:
        unique_spelling_errors.add(e['found'])

    error_counts = {
        "spelling":  len(unique_spelling_errors),
        "linebreak": len(linebreak_issues),
        "punct":     len(punct_errors),
        "maiyamok":  sum(1 for r in maiyamok_results if r['สถานะ'].startswith('❌')) + len(repeated_word_issues)
    }

    reasons = []
    if error_counts["linebreak"]:
        details = [f"{i['prev_part']} + {i['next_part']} → {i['combined']}" for i in linebreak_issues]
        reasons.append("พบการฉีกคำข้ามบรรทัด: " + "; ".join(details))
    if error_counts["spelling"] or dataset_errors:
        error_words   = list(set([e['word'] for e in spelling_errors_legit]))
        dataset_desc  = list(set([f"{e['wrong']} (ควรเป็น {e['right']})" for e in dataset_errors]))
        error_desc    = list(set([f"{e['found']} (ควรเป็น {e['should_be']})" for e in loanword_spell_errors]))
        reasons.append("ตรวจเจอคำสะกดผิด: " + ', '.join(error_words + dataset_desc + error_desc))
    if error_counts["punct"]:
        reasons.append(f"ใช้เครื่องหมายที่ไม่อนุญาต: {', '.join(punct_errors)}")
    if error_counts["maiyamok"]:
        all_maiyamok_errors = [
            f"{x['คำก่อนไม้ยมก']}: {x['สถานะ']}"
            for x in maiyamok_results if x['สถานะ'].startswith('❌')
        ] + repeated_word_issues
        reasons.append("ใช้ไม้ยมกผิด: " + '; '.join(all_maiyamok_errors))
    if not reasons:
        reasons.append("ไม่มีปัญหา")

    total_errors = sum(error_counts.values())
    if total_errors == 0:
        score = 2
    elif total_errors == 1:
        score = 1.5
    elif total_errors == 2:
        score = 1
    elif total_errors == 3:
        score = 0.5
    else:
        score = 0

    return score, reasons

# =========================================================
# S11 — Evaluate (Spelling + Linebreak + Punct + ไม้ยมก)
# =========================================================

def s11_evaluate(text):

    score, reasons = _evaluate_text_s11(str(text))

    return {
        "s11_score": score,
        "s11_reasons": reasons
    }

# =========================================================
# S12 — BERT Scoring (S12 model, TEXT_302)
# =========================================================

MAX_LENGTH_S12 = 416

def s12_predict_score(text):
    """S12: ส่งข้อความเข้า S12 model เพื่อทำนายคะแนน (single text)"""
    inputs = tokenizer_s12(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH_S12
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs    = model_s12(**inputs)
        pred_class = torch.argmax(outputs.logits, dim=1).item()
    return float(id2label_s12[pred_class])

def s12_predict_batch(texts):
    """S12: batch prediction ผ่าน Trainer"""
    df_tmp  = pd.DataFrame({"text": [str(t) for t in texts]})
    dataset = Dataset.from_pandas(df_tmp)

    def tokenize_fn(example):
        return tokenizer_s12(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH_S12
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer_s12)
    trainer       = Trainer(model=model_s12, data_collator=data_collator)
    predictions   = trainer.predict(dataset)
    preds         = np.argmax(predictions.predictions, axis=-1)
    return [float(id2label_s12[int(p)]) for p in preds]


# =========================================================
# S13 — BERT Scoring
# =========================================================

MAX_LENGTH_S13 = 416

def s13_predict_score(text):

    inputs = tokenizer_s13(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH_S13
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model_s13(**inputs)

        pred_class = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    return float(
        id2label_s13[pred_class]
    )


def s13_predict_batch(texts):

    df_tmp = pd.DataFrame({
        "text": [str(t) for t in texts]
    })

    dataset = Dataset.from_pandas(df_tmp)

    def tokenize_fn(example):

        return tokenizer_s13(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH_S13
        )

    dataset = dataset.map(
        tokenize_fn,
        batched=True
    )

    dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask"
        ]
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer_s13
    )

    trainer = Trainer(
        model=model_s13,
        data_collator=data_collator
    )

    predictions = trainer.predict(dataset)

    preds = np.argmax(
        predictions.predictions,
        axis=-1
    )

    return [
        float(id2label_s13[int(p)])
        for p in preds
    ]

# =========================================================
# COMBINED PIPELINE  (ข้อตกลง → S7 → S8 → S9 → S10 → S11 → S12)
# =========================================================

def score_student_answer(
    text_302,
    numline_302=None
):

    text_302 = str(text_302)

    # -------------------------
    # ข้อตกลง
    # -------------------------

    if numline_302 is not None:

        num_line = int(numline_302)

        if 1 <= num_line <= 2:
            numline_info = "คำตอบ 1-2 บรรทัด"
        else:
            numline_info = "ไม่ใช่คำตอบสั้น"

    else:
        num_line, numline_info = check_numline(
            text_302
        )

    has_keyword, keyword_info = (
        s7_keyword_classify(text_302)
    )

    s7_score = 1 if has_keyword else 0

    # -------------------------
    # ตรวจคัดลอก
    # -------------------------

    copy_result = check_copying(
        text_302
    )

    is_copy = copy_result["is_copy"]

    # =====================================================
    # CASE 1
    # ไม่มีคำบอกข้อคิดเห็น + คัดลอก
    # ได้ 0 ทั้งข้อ
    # =====================================================

    if (not has_keyword) and is_copy:

        total_score = 0

        return {
            "ข้อตกลง_บรรทัด": num_line,
            "ข้อตกลง_info": numline_info,

            **copy_result,

            "s7_score": 0,
            "s7_info": keyword_info,

            "s8_score": 0,
            "s9_score": 0,

            "s10_score": 0,
            "sentiment": "",
            "sentiment_th": "",
            "s10_mistakes": [],

            "s11_score": 0,
            "s11_reasons": ["คัดลอกบทอ่าน"],

            "s12_score": 0,
            "s13_score": 0,

            "TOTAL_SCORE": total_score
        }

    # =====================================================
    # CASE 2
    # มีคำบอกข้อคิดเห็น + คัดลอก
    # ตรวจเฉพาะ S7
    # =====================================================

    if has_keyword and is_copy:

        total_score = s7_score

        return {
            "ข้อตกลง_บรรทัด": num_line,
            "ข้อตกลง_info": numline_info,

            **copy_result,

            "s7_score": s7_score,
            "s7_info": keyword_info,

            "s8_score": 0,
            "s9_score": 0,

            "s10_score": 0,
            "sentiment": "",
            "sentiment_th": "",
            "s10_mistakes": [],

            "s11_score": 0,
            "s11_reasons": ["คัดลอกบทอ่าน"],

            "s12_score": 0,
            "s13_score": 0,

            "TOTAL_SCORE": total_score
        }

    # =====================================================
    # CASE 3
    # NUM_LINE 1-2
    # ตรวจเฉพาะ S7 S8
    # =====================================================

    if 1 <= num_line <= 2:

        s8_score = s8_predict_score(
            text_302
        )

        total_score = (
            s7_score +
            s8_score
        )

        return {
            "ข้อตกลง_บรรทัด": num_line,
            "ข้อตกลง_info": numline_info,

            **copy_result,

            "s7_score": s7_score,
            "s7_info": keyword_info,

            "s8_score": s8_score,

            "s9_score": 0,

            "s10_score": 0,
            "sentiment": "",
            "sentiment_th": "",
            "s10_mistakes": [],

            "s11_score": 0,
            "s11_reasons": ["ไม่ตรวจ"],

            "s12_score": 0,
            "s13_score": 0,

            "TOTAL_SCORE": total_score
        }

    # =====================================================
    # CASE 4
    # ตรวจปกติ
    # =====================================================

    s8_score = s8_predict_score(text_302)
    s9_score = s9_predict_score(text_302)

    s10_result = s10_evaluate(
        text_302
    )

    s11_result = s11_evaluate(
        text_302
    )

    s12_score = s12_predict_score(
        text_302
    )

    s13_score = s13_predict_score(
        text_302
    )

    # -------------------------
    # NUM_LINE 3-4
    # ลดครึ่ง
    # -------------------------

    if 3 <= num_line <= 4:

        s11_result["s11_score"] /= 2
        s12_score /= 2
        s13_score /= 2

        s11_result["s11_reasons"].append(
            "จำนวนบรรทัด 3-4 บรรทัด (ลดคะแนนครึ่งหนึ่ง)"
        )

    total_score = (
            s7_score +
            s8_score +
            s9_score +
            s10_result["s10_score"] +
            s11_result["s11_score"] +
            s12_score +
            s13_score
    )

    return {
        "ข้อตกลง_บรรทัด": num_line,
        "ข้อตกลง_info": numline_info,

        **copy_result,

        "s7_score": s7_score,
        "s7_info": keyword_info,

        "s8_score": s8_score,
        "s9_score": s9_score,

        **s10_result,
        **s11_result,

        "s12_score": s12_score,
        "s13_score": s13_score,

        "TOTAL_SCORE": total_score
    }
