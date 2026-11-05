# Lab Report: Assignment 2, Semantic Tagging with GloVe-50
AME 5151 Applied Linear Algebra Lab

---

### Submission contents
* `vec.py`: the `Vec` class from Assignment 1 (tuple storage, `__slots__`, `math.fsum` dot product)
* `glove_tagger.py`: main program (Parts A to H)
* `test_glove_tagger.py`: `unittest` test suite (11 tests, covering all 7 required cases)
* `manipal_linkedin.txt`: input text (three MAHE "Research in 60 Seconds" / ICHTMR 2026 LinkedIn posts, 320 words)
* `citations.md`: sources
* `program_output.txt`: captured output

### Running
```bash
pip install numpy gensim
export GLOVE_PATH=assets/glove50/glove_50_fast.wordvectors   # path to the GloVe-50 KeyedVectors file
python glove_tagger.py manipal_linkedin.txt
python -m unittest -v test_glove_tagger.py
```

---

## Part A: Tag vocabulary

```
research, innovation, education, university, students, faculty, campus,
engineering, medicine, public health, technology, artificial intelligence,
curriculum, collaboration, publication, scholarship, internship, startup,
alumni, careers
```

**Why these tags.** Manipal is a large multi-disciplinary university with strong medical, health sciences, engineering and IT schools, so the tags cover:
* core academic activity: `research, education, curriculum, publication, collaboration`
* the institution and its people: `university, campus, students, faculty, alumni`
* the main disciplines: `engineering, medicine, public health, technology, artificial intelligence`
* student outcomes and industry links: `internship, careers, startup, scholarship, innovation`

From the starter list I removed `laboratory`, `mentorship`, `entrepreneurship` and `accreditation`, which rarely appear in student posts. I replaced them with `public health`, `artificial intelligence`, `startup` and `careers`.

**Multi-word tags** (row of T = average of the component vectors):
* `public health` = ½(v_public + v_health)
* `artificial intelligence` = ½(v_artificial + v_intelligence)

All component words were checked against `model.key_to_index`.

## Part C: Preprocessing

Order: lowercase, strip punctuation, `split()`, remove stopwords, keep tokens with length ≥ 3, then split into in-vocabulary and OOV.

* Punctuation is removed with `str.translate`, mapping every `string.punctuation` character to a space. Because LinkedIn text often has curly quotes, en/em dashes, bullets and emojis, these are also mapped to spaces.
* **Stopwords added** to the handout's starter set: pronouns (`i, me, my, we, our, us, you, your, he, she, they, them, their, it, its`), demonstratives (`this, that, these, those`), auxiliaries and modals (`be, been, am, has, have, had, do, does, did, will, would, can, could, should, may, might, must`), and other function words (`as, into, about, over, after, before, during, between, but, so, than, then, also, very, just, not, no, all, any, each, every, some, such, more, most, other, what, which, who, when, where, why, how, here, there, up, out, off, again, once`).
* Repetitions are kept: each occurrence of a word adds a row to W.
* The posts open with headlines in Unicode "mathematical bold" letters (𝐖𝐡𝐚𝐭 𝐢𝐟 ...). Without normalisation every one of those words would be OOV, so `lowercase()` first applies `unicodedata.normalize("NFKC", ...)`, which maps them to plain letters.
* `http`, `https` and `www` are stopwords so that links do not add noise. Short-link codes such as `lnkd` and `pxdvsh` end up OOV.
* LinkedIn adds the word "hashtag" in front of every `#` when a post is copied. That artefact was removed from the text file.

## Parts D to G: Matrices and verification

* `T` and `W` are built as `float64` arrays (GloVe stores `float32`, so the vectors are upcast). This lets the NumPy and `Vec` results agree to machine precision, so a strict tolerance `atol=1e-9, rtol=0` is used in `np.allclose`.
* `similarity_matrix_vector_class` builds `Vec` objects and calls `cosine_similarity` for every (word, tag) pair with nested list comprehensions.
* `similarity_matrix_numpy` normalises rows with `np.sqrt(np.sum(M*M, axis=1))`, checks that no norm is zero, and returns `np.matmul(W_hat, T_hat.T)`.
* If no in-vocabulary words remain, the program exits with an error message before computing S.

## Results

```
Token statistics
  before preprocessing   : 320
  after preprocessing    : 217
  in-vocabulary          : 186
  out-of-vocabulary      : 31
  distinct OOV tokens    : anushree, behaviouraleconomics, biomedicalengineering, biomedicalresearch,
                           cognitivebias, consumerbehaviour, energyharvesting, etztrn6c, ichtmr, ichtmr2026,
                           inspiredbylife, karunakar, kotegar, lnkd, mahemanipal, maternalhealth, pxdvsh,
                           researchatmahe, researchinmahe, shyampradeep, tanavi, wearabletech

T.shape: (20, 50)
W.shape: (186, 50)
S.shape: (186, 20)
np.allclose(S_vector, S_numpy, atol=1e-09): True
np.max(np.abs(S_vector - S_numpy)): 4.44e-16

Rank  Tag            Score   Best-matching text word  Match
1     research       1.000   research                 same word
2     engineering    1.000   engineering              same word
3     technology     1.000   technology               same word
4     public health  0.947   health                   related word
5     students       0.882   student                  related word
6     medicine       0.872   medical                  related word
7     campus         0.847   school                   related word
8     university     0.826   school                   related word
```

Ranks 9 to 20:

| Tag | Score | Word |
|---|---|---|
| faculty | 0.802 | student |
| education | 0.801 | student |
| innovation | 0.788 | technology |
| collaboration | 0.767 | focuses |
| artificial intelligence | 0.759 | information |
| publication | 0.745 | edition |
| curriculum | 0.737 | interdisciplinary |
| scholarship | 0.704 | phd |
| careers | 0.672 | work |
| alumni | 0.639 | student |
| startup | 0.560 | marketing |
| internship | 0.559 | phd |

All 11 unit tests pass.

---

## Lab-report questions

**1.** With n in-vocabulary words: (a) a word vector is in ℝ⁵⁰, so it is 50 numbers (1×50 as a row); (b) W is n×50; (c) T is 20×50; (d) Tᵀ is 50×20; (e) S is n×20.

**2.** Let a = W_i and b = T_j. After normalisation the rows are a/‖a‖ and b/‖b‖. Their dot product is (a/‖a‖)·(b/‖b‖) = (a·b)/(‖a‖‖b‖), because the dot product is linear in each argument and the norms are scalars. This is the definition of cos(a, b). Normalisation does not change the direction of the vectors, only their length (now 1), and the cosine depends only on direction.

**3.** Matrix multiplication computes every row-of-the-left times column-of-the-right dot product. The columns of T̂ᵀ are the 20 normalised tag vectors, and the rows of Ŵ are the n normalised word vectors. So entry (i, j) of ŴT̂ᵀ is Ŵ_i · T̂_j = cos(W_i, T_j), and the n×20 result contains all 20n cosine similarities at once.

**4.** The score for `medicine` is the maximum of its column of S. The one word with very high similarity sets the score, and the low values for the other words do not lower it. So `medicine` would rank near the top even though only one word in the document supports it.

**5.** No. Max pooling is a "does at least one strong match exist?" measure. It ignores how many words relate to the tag. A tag that is mentioned once in passing gets the same 1.0 as a tag that the whole document is about. To judge the dominant topic you would need something like the mean of the column, a count of words above a threshold, or term frequency.

**6.** In my top eight, only three tags were matched by the **identical** word: `research`, `engineering` and `technology` (score 1). The other five were matched by a **different** word: `public health` by `health` (0.947), `students` by `student` (0.882), `medicine` by `medical` (0.872), and both `campus` and `university` by `school` (0.847 and 0.826).

The different-word matches show what the embeddings capture: words used in similar contexts in Wikipedia and news text. That includes inflections (`student`/`students`), adjective and noun forms of the same idea (`medical`/`medicine`), and category neighbours (`school` for both `campus` and `university`). What they don't capture is the actual meaning of the posts. `campus` and `university` rank high only because of the word "school" in "Manipal School of Information Sciences", and the posts are not about campus life. `artificial intelligence` (0.759) matched `information` for the same reason. A multi-word tag like `public health` also scores well against `health` just because "health" is one of its two averaged components. Exact matches always win with 1.0, even when the word is a single mention. Cosine similarity measures word-level relatedness, not topic relevance or meaning in context.

**7.** No. The output can only contain tags that are rows of T. Max pooling reorders the existing 20 tags and cannot create a column for an unseen concept. A missing concept (e.g. `sports` or `rankings`) is either ignored or leaks into the score of the closest existing tag. Adding it would require editing the tag list and rebuilding T.

**8.** OOV tokens have no GloVe vector, so they have no row to contribute to W. Substituting a zero vector would make ‖W_i‖ = 0 and the cosine undefined (division by zero). A random or averaged vector would add similarities that do not come from the text. Excluding them, and reporting them, keeps every row of S meaningful.

**9.** The `Vec` loop and the NumPy product compute the same matrix entry by entry: S_ij = (W_i · T_j)/(‖W_i‖‖T_j‖). The loop evaluates this formula separately for every pair (computing norms repeatedly). The NumPy version factors the formula: S = D_W⁻¹ W Tᵀ D_T⁻¹, where D_W and D_T are diagonal matrices of row norms. That is the same as (D_W⁻¹W)(D_T⁻¹T)ᵀ = ŴT̂ᵀ. Since both use float64 here, the maximum difference was about 4.4×10⁻¹⁶, which is rounding error only.
