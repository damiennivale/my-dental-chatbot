# Evaluation summary (k = 5)

## 1. Retrieval accuracy (answerable questions)

| Group | Questions | Doc Recall@5 | Page Recall@5 | Doc Precision@5 | Page Precision@5 | MRR (page) |
|---|---|---|---|---|---|---|
| **All** | 49 | 0.980 | 0.776 | 0.796 | 0.192 | 0.628 |
| professional mode | 31 | 0.968 | 0.774 | 0.839 | 0.200 | 0.675 |
| patient mode | 18 | 1.000 | 0.778 | 0.722 | 0.178 | 0.546 |
| CLINICAL_PREVENTION_PROGRAMME_FOR_CARIES_2020.pdf | 4 | 0.750 | 0.250 | 0.250 | 0.050 | 0.250 |
| CPG_Management_of_Early_Childhood_Caries__3rd_Edition__ | 7 | 1.000 | 0.714 | 0.743 | 0.200 | 0.548 |
| CPG_Management_of_Periodontal_Abscess__2nd_Edition__201 | 4 | 1.000 | 1.000 | 1.000 | 0.250 | 1.000 |
| FLUORIDE_VARNISH_PROGRAMME_FOR_TODDLERS_2019.pdf | 4 | 1.000 | 0.500 | 0.900 | 0.100 | 0.375 |
| Guidelines_A_School_Based_Fissure_Sealant_Programme_200 | 2 | 1.000 | 1.000 | 0.500 | 0.200 | 0.292 |
| Guidelines_Early_Childhood_Oral_Healthcare_2008.pdf | 5 | 1.000 | 0.800 | 0.600 | 0.240 | 0.700 |
| Oral_Healthcare_For_Young_Adults_in_Malaysia_2019.pdf | 4 | 1.000 | 1.000 | 0.900 | 0.300 | 1.000 |
| Oral_Healthcare_for_Elderly_in_Malaysia_2002.pdf | 4 | 1.000 | 1.000 | 1.000 | 0.200 | 0.750 |
| Oral_Healthcare_for_PreSchool_Children_in_Malaysia_2019 | 3 | 1.000 | 0.333 | 0.733 | 0.133 | 0.333 |
| Oral_Healthcare_for_Pregnant_Mother_in_Malaysia_2004.pd | 5 | 1.000 | 1.000 | 0.880 | 0.200 | 0.750 |
| Primary_Prevention_and_Early_Detection_of_Oral_Potentia | 3 | 1.000 | 1.000 | 1.000 | 0.267 | 1.000 |
| School_Based_Fluoride_Mouth_Rinsing_Programme_2023.pdf | 4 | 1.000 | 0.750 | 1.000 | 0.150 | 0.396 |

Retrieval time per question: mean 0.149s, 95th percentile 0.080s.

(Retrieval accuracy above is measured with the similarity gate switched OFF, so it reflects the search itself. The gate is evaluated in section 2.)

## 2. Similarity gate (MIN_DENSE_SIM) and unanswerable questions

Current MIN_DENSE_SIM = 0.45. Valid questions wrongly rejected: 2 of 49 (Q27, Q49). Unanswerable questions rejected: 5 of 6 (not rejected: N04).

| MIN_DENSE_SIM | Valid questions wrongly rejected | Unanswerable questions rejected |
|---|---|---|
| 0.30 | 0 of 49 (0.0%) | 1 of 6 |
| 0.35 | 0 of 49 (0.0%) | 2 of 6 |
| 0.40 | 0 of 49 (0.0%) | 4 of 6 |
| 0.45 | 2 of 49 (4.1%) | 5 of 6 |
| 0.50 | 6 of 49 (12.2%) | 5 of 6 |
| 0.55 | 10 of 49 (20.4%) | 6 of 6 |
| 0.60 | 27 of 49 (55.1%) | 6 of 6 |

Lowest gate value among valid questions: 0.420. A threshold of about 0.41 wrongly rejects none of them and rejects 5 of 6 unanswerable questions.
The gate is only a safety net: the prompt's 'not covered by the guidelines' rule handles the unanswerable questions that get through.

## 3. Guardrails

| Type | Passed | Total |
|---|---|---|
| emergency | 8 | 8 |
| injection | 7 | 7 |
| out_of_scope | 7 | 7 |
| valid questions (must pass) | 55 | 55 |
