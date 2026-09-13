# External reviewer findings

Tracking issue: https://github.com/kwojdalski/masters_thesis/issues/888

Tracking issue for the external reviewer report by **dr hab. Robert Slepaczuk, prof. UW**, dated 2026-09-12.

**Result: 48/59, grade 5 (very good).** Content 37/45, form 11/14. The reviewer also ticked **YES** for submission to the A. Semkow competition for the best graduate thesis.

Full report: `docs/masters_thesis/reviews/2026-09-12-slepaczuk-reviewer-report.pdf` (text extract alongside it).
Work individual findings with the `quant-reviewer` agent.

### Where the points were lost

| Section | Score | Lost |
| --- | --- | --- |
| Motivation | 4/5 | nearest competing work not cited |
| Research question / hypotheses | 4/5 | no null, alternative, significance level or test statistic |
| Method (choice) | 4/5 | single run per configuration |
| Method (use) | 4/5 | no statistical inference reported, no non-RL baseline |
| Knowledge | 4/5 | Markov property misdefined, horizons unreconciled |
| Literature (selection) | 4/5 | three misattributions |
| Literature (usage) | 5/5 | — |
| Results | 4/5 | three figures do not reconcile |
| Logic | 4/5 | conclusion overclaims against 6.1.8 |
| Titles | 2/2 | — |
| Construction | 2/2 | — |
| Introduction / conclusions | 2/2 | — |
| **References** | **1/2** | missing entries, misattributions, formatting |
| **Edition** | **1/2** | Table 4 conventions, figure labelling |
| Language | 2/2 | — |
| **Terminology** | **1/2** | Markov, Kyle's lambda, VPIN, notation collisions |

### Results reconciliation (the reviewer's three defence questions)

- [ ] #855 Win+Lose sums differ across policies at zero fee and should not
- [ ] #856 21,776 position changes does not reconcile with 0.53 turnover (factor ~45)
- [ ] #858 Extend the bid/ask repricing from one instrument to all six using Table 13
- [ ] #857 Figure 4 shows TD3 below zero cumulative reward while near the top in Figure 3, and is never interpreted

### Method and evidence (experimental — cost these before running)

- [ ] #859 Every headline result rests on a single run; rerun with at least 10 seeds
- [ ] #860 No statistical inference reported in Chapter 6 although the tests were implemented
- [ ] #861 Add a linear baseline on the same 10 features to Table 4
- [ ] #862 Table 4 presents unweighted six-instrument averages as portfolio statistics
- [ ] #863 Normalisation pipeline described two different ways in 5.3.3 and Figure 6
- [ ] #887 Four trading days with test confined to half a session leaves no room for robustness checks

### Hypotheses and argument

- [ ] #864 None of the four hypotheses has a null, alternative, significance level or test statistic
- [ ] #865 Hypothesis 1 asserts two different claims in one sentence
- [ ] #866 Hypothesis 1 discloses its own outcome in the Introduction
- [ ] #867 Conclusion says the agent beats passive benchmarks; 6.1.8 declines that claim

### Definitions, terminology and notation

- [ ] #868 Markov property defined as a property of the optimal action
- [ ] #869 Kyle's lambda printed with the single-auction coefficient, attributed to the continuous-time model
- [ ] #870 The feature named hft_vpin is not VPIN as the glossary defines it
- [ ] #871 Signed-trade-flow credited to Lee & Ready (1991) without applying that algorithm
- [ ] #883 phi denotes the actor for TD3/DDPG but the value network for PPO
- [ ] #884 The object priced by the simulator carries four different names
- [ ] #885 Algorithm 1 says "Load L3 order book dataset" for MBP-10 data

### Knowledge and diagnosis

- [ ] #872 Reconcile the four time horizons in play
- [ ] #873 Report the update-to-data ratio implied by Table 10 given the action saturation

### References

- [ ] #874 Anchor the novelty claim to Nevmyvaka/Feng/Kearns (2006) and Sirignano & Cont (2019)
- [ ] #875 At least eight sources relied on in the text have no bibliography entry
- [ ] #876 At least four attributions misrepresent their sources (Mnih et al. reward clipping)
- [ ] #877 Bibliography formatting: incomplete entries and a broken van Hasselt sort key

### Edition, construction and language

- [ ] #878 Table 4 omits a buy-and-hold row although 6.1.1 calls it the primary economic benchmark
- [ ] #879 Three figures need fuller axis labelling (Figures 3, 4 and 5)
- [ ] #880 Feature-selection scoring formulas sit in Appendix J but are needed for Section 4.2.5
- [ ] #881 The six-fold smaller training budget disclosure sits in the wrong section
- [ ] #882 Mixed US and UK spelling throughout the document
- [ ] #886 Run-on sentence at p.67 and a Polish inflection error in the title-page AI declaration

### Overlaps with existing internal findings

Several reviewer findings independently confirm what the internal auditors already logged. Fix once, close both:

| Reviewer issue | Existing issue |
| --- | --- |
| #866 | #837 |
| #867 | #829, #795 |
| #878 | #835 |
| #881 | #834, #843 |
| #859 | #763, #778 |
| #856 | #828 |
| #872 | #833 |
| #885 | #852 |
| #870 | #853 |
