# Recommendation System Data Score Curation for LLMs


This project applies the DS2 (ICLR 2025) framework for data score curation to recommendation systems, evaluating whether the approach generalizes effectively beyond its original setting.

Google Search Team — LLM Data Augmentation Project
Manager: Shuang Yang
Mentor: Jingjing Liu

------ 

## 🎉🎉 Progress 
- [x] [2026.01.26] 🚀🚀 Adjust the code from [**DS2**](https://github.com/UCSC-REAL/DS2).


### 🧩 Step 1. Raw Score Error Detection
One can execute the score error detection by running
```
python diagnose.py
```
The corresponding curation report files can be found in the path `score_curation_results/`.


---

### 🧩 Step 3. Score Curation
Given the generated score curation reports, one can directly obtain the curated score by 
```
python score_generation.py
``` 
The generated scores are encoded into original dataset, using `keyword`: `curated_score`, `diversity_score`, and their combined score `final_curated_score`.

