"""Resampling counts and the error rate every benchmark shares, named once so no copy can drift."""

#: Resamples behind a BCa interval, the count the published intervals were computed at.
DEFAULT_N_RESAMPLES = 1000

#: Draws behind a permutation test of one statistic.
DEFAULT_N_PERMUTATIONS = 10000

#: Draws behind a joint one-vs-rest maxT test, which rebuilds the null once per group.
DEFAULT_N_GROUP_PERMUTATIONS = 2000

#: Family-wise error rate for every interval and correction in the project.
DEFAULT_ALPHA = 0.05
