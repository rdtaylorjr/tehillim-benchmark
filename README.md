# tehillim-benchmark

## Overview

This repository evaluates representations of Biblical Hebrew against two fixed Psalms annotations and a related trajectory analysis. It scores half-verse representations for their capacity to rank annotated parallel relationships, scores psalm representations for their capacity to separate a supplied genre classification, and tests whether ordered within-psalm representation profiles covary with that classification. It is the evaluation layer of the Tehillim project. Representation construction is in [tehillim-embeddings](https://github.com/rdtaylorjr/tehillim-embeddings) and generated outputs are in [tehillim-data](https://github.com/rdtaylorjr/tehillim-data).

## Data

The parallelism benchmark reads `parallel_*` Text-Fabric features released by [tehillim-logos](https://github.com/rdtaylorjr/tehillim-logos) on ETCBC BHSA `half_verse` nodes. The features encode aligned group membership, type, signature, member span, and an ambiguity flag. They derive from Logos Bible Software's *Psalms Explorer Dataset*, used with permission. A separate runtime CSV assigns one of seven source genres to each of the 150 psalms. Neither licensed annotation source is committed here.

The repository reads dense and sparse Parquet vectors from the BHSA `half_verse` scope, `corpus=bhsa/unit=half_verse/`, of `tehillim-embeddings`. Its node identifiers match the parallelism, genre, and trajectory labels. It writes CSV, Parquet, and JSON outputs to `tehillim-data` under `analysis=benchmark/benchmark={parallelism,genre,trajectory}/domain={...}/stage={...}`. This division keeps code, source annotation, representations, and derived results independently inspectable.

The data are operational annotations rather than neutral descriptions of poetic form. Logos supplies neither an annotation protocol nor adjudication history nor inter-annotator reliability estimate. BHSA `half_verse` boundaries and its morphosyntactic features also embody ETCBC analytic decisions. The benchmark preserves these choices as data conditions and reports their consequences without treating them as settled linguistic categories.

## Methodology

The loader reconstructs each annotation group from the `parallel_*` features. In a signature, a dash separates lines and a letter marks a member. Repeated letters mark members that correspond. The loader pairs every two members sharing a letter, whether across lines or within one. When no letter repeats, it pairs adjacent lines as wholes, taking the union of each line's members. A dashless signature pairs adjacent members.

A pair is excluded when a member is unresolved or marked ambiguous, or when both sides resolve to the same half-verse set. Pairs within a group that resolve to the same node sets collapse to one. This projects a literary annotation onto a finite set of reproducible retrieval observations while preserving an audit trail for omitted cases.

Average Precision is the primary outcome. Cosine similarity ranks annotated relations against adjacent, within-psalm half-verse pairs excluding nodes in surviving retrieval pairs. This local control retains generic adjacency and topical continuity. It does not create an unannotated nonparallel background because the source annotation covers most nodes. AUC, rank-based retrieval measures, similarity calibration against unmarked background nodes, and type-specific summaries remain secondary descriptions. Average Precision is reported with its positive-class prevalence because its scale depends on the true-to-control ratio.

Confidence intervals use a psalm-clustered BCa bootstrap. Resampling whole psalms retains the dependence among relations drawn from the same poem. Genre confidence intervals use a vertex bootstrap that resamples psalms and reconstructs their derived pair population. The code applies Benjamini-Hochberg and Benjamini-Yekutieli adjustments within defined metric, source, and scope families. Per-genre permutation tests shuffle psalm labels and use a joint maxT null across genres. These procedures test evidence against the stated labels. They do not decide whether a representation has captured parallelism or genre as literary phenomena.

When a vertex bootstrap resamples the same original psalm more than once, pairs between those repeated draws are excluded before scoring. Such pairs are constructed positives with cosine similarity `1.0`, rather than observed psalm pairs.

For genre discrimination, each psalm vector is the mean of its available half-verse vectors. The evaluator scores all 11,175 unordered psalm pairs, labeling a pair positive when both psalms share the supplied source genre. It reports pooled and one-versus-rest Average Precision and AUC. The unequal class sizes make pooled outcomes largely responsive to prevalent genres, so genre-specific results remain necessary.

Trajectory analysis retains half-verse order. It derives a psalm centroid, an ordered cosine self-similarity matrix, adjacent similarity, step magnitude, and turning angle. Structural profiles are compared after length-normalized resampling or dynamic-time-warping alignment. Permutation tests compare within- and between-genre distances before and after residualizing distance on length difference, then on length difference and content distance. These controls identify whether an association persists under those specified nuisance models. They do not supply a theory-free separation of form from meaning.

## Results

The public interface payloads contain 148 parallelism variants and 222 genre variants, excluding order-shuffle draws. An earlier result-store summary reported 222 parallelism and 220 genre variants. The repository has no release manifest that reconciles those output versions.

The published parallelism maxima were calculated under a superseded pairing rule that produced 1,110 relations, 755 within one annotated line. They are retained in the result store for provenance and are not reported here as findings for the current benchmark. The current rule produces 2,392 relations from 2,000 of 2,292 source groups. The current relation set therefore requires a versioned public release before its representation scores can be reported or compared.

| Relation-construction audit | Count |
| --- | ---: |
| Source groups | 2,292 |
| Groups yielding one or more relations | 2,000 |
| Candidate pairs | 3,774 |
| Candidate pairs with an ambiguous member | 38 |
| Candidate pairs with a missing member | 6 |
| Candidate pairs resolving to one half-verse set | 557 |
| Collapsed element pairs duplicating a retained line pair | 781 |
| Current retrieval relations | 2,392 |

Of the 557 candidate pairs resolving to one half-verse set, 129 are line pairs whose two annotated lines fall within one accentual half-verse and 428 are element pairs within one half-verse. BHSA half-verse boundaries coincide with a clause boundary in 95.1 percent of cases, against a 28.7 percent base rate. This is descriptive agreement between two segmentations. It does not validate either segmentation or identify a poetic line independently of the annotation source. The local control includes an annotation-bearing node in 2,720 pairs. Only 64 adjacent bicola carry no `parallel_*` annotation. Genre outcomes use the 11,175 unordered psalm pairs.

The result store records a recurring genre result: all 43 semantic variants place the supplied Hymn class below AUC 0.5, with a mean AUC of 0.385 and a maximum of 0.469. This outcome warrants inspection of the source labels, class composition, psalm length, and representation behavior. It does not establish that hymns lack a coherent literary profile.

## Limitations

Parallelism and genre scores quantify agreement with one commercial annotation resource. Source agreement does not validate a reference account of Hebrew poetry. A single genre per psalm suppresses mixed forms, diachronic relations, and the possibility that source categories overlap. The treatment of Psalms 57, 60, and 108 illustrates this constraint because Psalm 108 combines material from texts assigned another source category.

The adjacent control excludes scored retrieval pairs, yet it cannot match every source of lexical, grammatical, topical, or positional dependence. It also cannot establish a contrast with nonparallel text because only 64 adjacent bicola lack the source annotation. Scores are pooled across BHSA text types. Since text type is a syntactic analysis that can covary with the supplied genre labels, a pooled genre result cannot distinguish genre association from text-type composition. Representation choices and reported maxima use the same annotations and corpus. They are exploratory comparisons, without a held-out psalm partition or a null that reassigns annotation groups under within-psalm constraints. Psalm-level resampling addresses clustering at that level while leaving dependence within annotation groups and between textual relatives. Trajectory residualization depends on linear nuisance models and observable covariates. Remote embedding services and evolving model checkpoints can also change an input representation without changing this code.

A future confirmatory pass should fix representation choices on a predetermined psalm partition, evaluate them on a separate partition, stratify or condition on text type, and compare observed scores with within-psalm annotation reassignments that preserve documented group conditions. These procedures can test a stated representation claim. They cannot resolve the source taxonomy or interpret a poem.

## Reproducibility

Python 3.10 or later and the dependencies in `pyproject.toml` are required. The BHSA checkout is pinned to `v1.8.1` and the Logos Text-Fabric module to `v1.0`. Unit tests run without licensed annotations or vector files. Integration runs require permitted access to the Logos features, the source genre CSV, a local embeddings checkout, and a writable data checkout. The checked-in public payloads do not carry a complete manifest of their inputs or source revision. The result-version discrepancy above shows why a release manifest is necessary. Byte-identical reproduction also depends on the same external annotation and model artifacts.

A Snakemake driver coordinates declared scoring and interface-export steps and writes manifests for
the outputs it creates.

Sparse `morph_signature` trigram partitions require the library's sparse evaluation path. The standard batch commands do not dispatch to that path.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
./check.sh
```

## Usage

Run a benchmark script with an embeddings directory and a matching output directory in a local
`tehillim-data` checkout:

```bash
.venv/bin/python -m genre.scripts.compare_models \
  /path/to/genre-labels.csv \
  /path/to/tehillim-embeddings/data \
  --output /path/to/tehillim-data/analysis=benchmark/benchmark=genre/domain=semantic/stage=raw/summary.csv
```

## References

Benjamini, Yoav, and Yosef Hochberg. 1995. [“Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.”](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) *Journal of the Royal Statistical Society: Series B* 57.1: 289-300.

Benjamini, Yoav, and Daniel Yekutieli. 2001. [“The Control of the False Discovery Rate in Multiple Testing under Dependency.”](https://doi.org/10.1214/aos/1013699998) *Annals of Statistics* 29.4: 1165-1188.

Efron, Bradley. 1987. [“Better Bootstrap Confidence Intervals.”](https://doi.org/10.1080/01621459.1987.10478410) *Journal of the American Statistical Association* 82.397: 171-185.

de la Selle, Théotime, and Laurence Mellerin. [“Detection and Typology of Psalmic Text Reuses in the New Testament.”](https://doi.org/10.3390/rel17010088) *Religions* 17, no. 1 (2026): 88.

Davis, Jesse, and Mark Goadrich. 2006. [“The Relationship Between Precision-Recall and ROC Curves.”](https://doi.org/10.1145/1143844.1143874) In *Proceedings of the 23rd International Conference on Machine Learning*, 233-240. ACM.

Gillmayr-Bucher, Susanne. [“Relecture of Biblical Psalms: A Computer Aided Analysis of Textual Relations Based on Semantic Domains.”](https://doi.org/10.1163/9789004493339_021) Pages 309-321 in *Bible and Computer: The Stellenbosch AIBI-6 Conference*. Leiden: Brill, 2002.

Montaner, Luis Vegas. “Masoretic Tradition and Syntactic Analysis of the Psalms.” Pages 317-335 in *Tradition and Innovation in Biblical Interpretation: Studies Presented to Professor Eep Talstra on the Occasion of His Sixty-Fifth Birthday*, 2011.

Muennighoff, Niklas, Nouamane Tazi, Loic Magne, and Nils Reimers. 2023. [“MTEB: Massive Text Embedding Benchmark.”](https://aclanthology.org/2023.eacl-main.148/) In *Proceedings of EACL 2023*, 2014-2037.

Naaijer, Martijn, and Dirk Roorda. [“Parallel Texts in the Hebrew Bible, New Methods and Visualizations.”](https://doi.org/10.48550/arXiv.1603.01541) 2016.

Phipson, Belinda, and Gordon K. Smyth. 2010. [“Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn.”](https://doi.org/10.2202/1544-6115.1585) *Statistical Applications in Genetics and Molecular Biology* 9.1.

Roorda, Dirk, Christiaan Erwich, Cody Kingham, and SeHoon Park. 2023. [*ETCBC/bhsa*](https://github.com/ETCBC/bhsa).

Sakoe, Hiroaki, and Seibi Chiba. 1978. [“Dynamic Programming Algorithm Optimization for Spoken Word Recognition.”](https://doi.org/10.1109/TASSP.1978.1163055) *IEEE Transactions on Acoustics, Speech, and Signal Processing* 26.1: 43-49.

Smiley, David M. [“Intertextual Parallel Detection in Biblical Hebrew: A Transformer-Based Benchmark.”](https://doi.org/10.48550/arXiv.2506.24117) 2025.

Westfall, Peter H., and S. Stanley Young. 1993. *Resampling-Based Multiple Testing: Examples and Methods for P-Value Adjustment*. Wiley.

Winkler, Anderson M., Gerard R. Ridgway, Matthew A. Webster, Stephen M. Smith, and Thomas E. Nichols. 2014. [“Permutation Inference for the General Linear Model.”](https://doi.org/10.1016/j.neuroimage.2014.01.060) *NeuroImage* 92: 381-397.

## License

MIT. The Logos annotation source and BHSA data have separate terms of use.
