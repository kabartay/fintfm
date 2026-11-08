# DR4.2 licence half: CC18 binary and multiclass candidates read at source (2026-11-08)

Draft. Evaluation use only. Nothing here edits `tasks.md` or `SOURCE_LICENCES`; the lead merges.

## Result

- 57 OpenML-CC18 binary and multiclass datasets pass the overlap filters (TabArena-v0.1 and BeyondArena by id, name, same source): 24 binary and 33 multiclass.
- Verdicts: 44 PASS, 0 FAIL, 13 UNVERIFIED.
- PASS: 18 of 24 binary, 26 of 33 multiclass. UNVERIFIED: 6 binary, 7 multiclass. No candidate carries a stated non-commercial or copyleft term, so FAIL is empty.
- Target mix (about 30 binary / 8 multiclass): multiclass is met (26 PASS candidates for 8 slots). Binary is not: 18 of 30, of which 16 are within the 136-feature cap (madelon, 500 features, and Internet-Advertisements, 1,558, exceed it). No further CC18 binary task is left to add; even the 6 UNVERIFIED binary tasks would reach only 24.
- If every PASS below is accepted, the result reproduces the binary and multiclass part of the suite already in `proxy_suite.py` (18 binary, 8 multiclass). I re-read its 2026-10-16 UCI, Zenodo, Kaggle and GitHub entries and found no change.

## Doubt for the owner: the UCI statement is retroactive

All 36 UCI dataset pages (API and page read on 2026-11-08) carry the same sentence:

> "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license. This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given."

None lacks it. But every one of the 36 was donated before that policy existed (the latest donation shown is 2016-08-31; three pages show no date), and UCI's donation policy (https://archive.ics.uci.edu/contribute/donation) is written for future donors: "Datasets will be licensed under a Creative Commons Attribution 4.0 International license (CC BY 4.0)". It says nothing on earlier donations, and I found no UCI announcement relicensing them. The page statement is therefore a blanket statement by UCI, not evidence of a grant by the original donor or collector.

- Page-statement reading: 18 binary and 26 multiclass pass.
- Strict reading (a donation after the change, or the donor's own terms): only pc1 (Zenodo, Menzies), numerai28.6 (Kaggle, Numerai) and Fashion-MNIST (MIT, Zalando) pass, that is 2 binary and 1 multiclass.

The owner decides which reading applies. UCI rows carry their donation date so either can be applied.

Other doubts:

- Adult (UCI 2) is a US Census Bureau extract; the page states CC BY 4.0 and no conflict was found.
- Devanagari (UCI 389): the page states CC BY 4.0; OpenML points to the creators' own site and a Kaggle mirror, which I did not read.
- Fashion-MNIST: MIT covers the creator's repository; the images are Zalando article photographs. Not checked further.
- numerai28.6: Kaggle's field says CC0; the uploader is the data owner.
- UNVERIFIED rows: a 2026-11-08 re-read of the source pages found no licence text on any of them. pc4, pc3, kc2 and method-level kc1 have no Zenodo record (Zenodo holds pc1, pc5, kc3 under Menzies). I did not reach the ELENA project page (phoneme, texture) or an original donor page for eucalyptus, electricity, the analcatdata sets, MNIST or CIFAR-10. "Not found" is not "none exists".
- Method: UCI pages fetched and the licence sentence extracted from the page text; no dataset contents downloaded. A licence on a mirror (OpenML, Kaggle, Hugging Face) was not used, except where the Kaggle uploader is the data owner (numerai).

## Counts

| Type | Candidates | PASS | FAIL | UNVERIFIED |
| --- | --- | --- | --- | --- |
| binary | 24 | 18 (16 UCI, 1 Zenodo, 1 Kaggle) | 0 | 6 |
| multiclass | 33 | 26 (25 UCI, 1 GitHub MIT) | 0 | 7 |
| total | 57 | 44 | 0 | 13 |

Multiclass PASS tasks wider than the 136-feature cap: mfeat-factors (216), mfeat-pixel (240), semeion (256), cnae-9 (856), isolet (617), har (561), Fashion-MNIST (784), Devnagari-Script (1,024). Five of the eight tasks the lowest-id rule picks are Multiple Features views (UCI 72), as before.

## Table

For every UCI row the licence column reads "UCI CC BY 4.0 sentence": the sentence quoted above, found on that dataset's own page on 2026-11-08. The Note column gives the UCI donation date shown on the page. Verdict rule: PASS for CC BY, CC0, public domain, MIT or an explicit licence allowing commercial use; UNVERIFIED when the source states none.

### Binary

| OpenML data id | Task | Name | Rows x features | Original source | Licence or terms as stated | Verdict | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 3 | kr-vs-kp | 3,196 x 36 | https://archive.ics.uci.edu/dataset/22/chess+king+rook+vs+king+pawn | UCI CC BY 4.0 sentence | PASS | UCI donated 1989-07-31 |
| 15 | 15 | breast-w | 699 x 9 | https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original | UCI CC BY 4.0 sentence | PASS | UCI donated 1992-07-14 |
| 44 | 43 | spambase | 4,601 x 57 | https://archive.ics.uci.edu/dataset/94/spambase | UCI CC BY 4.0 sentence | PASS | UCI donated 1999-06-30 |
| 50 | 49 | tic-tac-toe | 958 x 9 | https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame | UCI CC BY 4.0 sentence | PASS | UCI donated 1991-08-18 |
| 151 | 219 | electricity | 45,312 x 8 | http://www.inescporto.pt/~jgama/ales/ales_5.html | none found (page 404 (10-16); MOA datasets page 2026-11-08 has no licence text) | UNVERIFIED |  |
| 1049 | 3902 | pc4 | 1,458 x 37 | https://openscience.us/repo/defect/mccabehalsted/pc4.html | none found (no licence; Zenodo search 2026-11-08 finds only pc1, pc5, kc3 under Menzies) | UNVERIFIED |  |
| 1050 | 3903 | pc3 | 1,563 x 37 | https://openscience.us/repo/defect/mccabehalsted/pc3.html | none found (no licence; Zenodo search 2026-11-08 finds only pc1, pc5, kc3 under Menzies) | UNVERIFIED |  |
| 1063 | 3913 | kc2 | 522 x 21 | https://openscience.us/repo/defect/mccabehalsted/kc2.html | none found (no licence; Zenodo search 2026-11-08 finds only pc1, pc5, kc3 under Menzies) | UNVERIFIED |  |
| 1067 | 3917 | kc1 | 2,109 x 21 | https://openscience.us/repo/defect/mccabehalsted/kc1.html | none found (Zenodo 268441 (CC BY 4.0) is the 94-variable class-level KC1, not this 21-feature method-level kc1) | UNVERIFIED |  |
| 1068 | 3918 | pc1 | 1,109 x 21 | https://zenodo.org/records/268456 | Zenodo licence field: "cc-by-4.0" (record "pc1", creator Tim Menzies, the donor of the NASA MDP sets) | PASS |  |
| 1590 | 7592 | adult | 48,842 x 14 | https://archive.ics.uci.edu/dataset/2/adult | UCI CC BY 4.0 sentence | PASS | UCI donated 1996-04-30 |
| 1510 | 9946 | wdbc | 569 x 30 | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic | UCI CC BY 4.0 sentence | PASS | UCI donated 1995-10-31 |
| 1489 | 9952 | phoneme | 5,404 x 5 | https://sci2s.ugr.es/keel/dataset.php?cod=105 | none found (KEEL page re-read 2026-11-08, no licence; ELENA origin page not reached) | UNVERIFIED |  |
| 1485 | 9976 | madelon | 2,600 x 500 | https://archive.ics.uci.edu/dataset/171/madelon | UCI CC BY 4.0 sentence | PASS | UCI donated 2008-02-28 |
| 1486 | 9977 | nomao | 34,465 x 118 | https://archive.ics.uci.edu/dataset/227/nomao | UCI CC BY 4.0 sentence | PASS | UCI donated 2012-07-03 |
| 1487 | 9978 | ozone-level-8hr | 2,534 x 72 | https://archive.ics.uci.edu/dataset/172/ozone+level+detection | UCI CC BY 4.0 sentence | PASS | UCI donated not shown |
| 1462 | 10093 | banknote-authentication | 1,372 x 4 | https://archive.ics.uci.edu/dataset/267/banknote+authentication | UCI CC BY 4.0 sentence | PASS | UCI donated 2013-04-15 |
| 4534 | 14952 | PhishingWebsites | 11,055 x 30 | https://archive.ics.uci.edu/dataset/327/phishing+websites | UCI CC BY 4.0 sentence | PASS | UCI donated 2015-03-25 |
| 6332 | 14954 | cylinder-bands | 540 x 39 | https://archive.ics.uci.edu/dataset/32/cylinder+bands | UCI CC BY 4.0 sentence | PASS | UCI donated 1995-07-31 |
| 23381 | 125920 | dresses-sales | 500 x 12 | https://archive.ics.uci.edu/dataset/289/dresses+attribute+sales | UCI CC BY 4.0 sentence | PASS | UCI donated 2014-02-18 |
| 40994 | 146819 | climate-model-simulation-crashes | 540 x 20 | https://archive.ics.uci.edu/dataset/252/climate+model+simulation+crashes | UCI CC BY 4.0 sentence | PASS | UCI donated 2013-06-17 |
| 40983 | 146820 | wilt | 4,839 x 5 | https://archive.ics.uci.edu/dataset/285/wilt | UCI CC BY 4.0 sentence | PASS | UCI donated 2014-03-12 |
| 23517 | 167120 | numerai28.6 | 96,320 x 21 | https://www.kaggle.com/datasets/numerai/encrypted-stock-market-data-from-numerai | Kaggle licence field: "CC0: Public Domain" (uploader Numerai, the data owner) | PASS |  |
| 40978 | 167125 | Internet-Advertisements | 3,279 x 1558 | https://archive.ics.uci.edu/dataset/51/internet+advertisements | UCI CC BY 4.0 sentence | PASS | UCI donated 1998-06-30 |

### Multiclass

| OpenML data id | Task | Name | Rows x features | Original source | Licence or terms as stated | Verdict | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 6 | letter | 20,000 x 16 | https://archive.ics.uci.edu/dataset/59/letter+recognition | UCI CC BY 4.0 sentence | PASS | UCI donated 1990-12-31 |
| 11 | 11 | balance-scale | 625 x 4 | https://archive.ics.uci.edu/dataset/12/balance+scale | UCI CC BY 4.0 sentence | PASS | UCI donated 1994-04-21 |
| 12 | 12 | mfeat-factors | 2,000 x 216 | https://archive.ics.uci.edu/dataset/72/multiple+features | UCI CC BY 4.0 sentence | PASS | UCI donated not shown |
| 14 | 14 | mfeat-fourier | 2,000 x 76 | https://archive.ics.uci.edu/dataset/72/multiple+features | UCI CC BY 4.0 sentence | PASS | UCI donated not shown |
| 16 | 16 | mfeat-karhunen | 2,000 x 64 | https://archive.ics.uci.edu/dataset/72/multiple+features | UCI CC BY 4.0 sentence | PASS | UCI donated not shown |
| 18 | 18 | mfeat-morphological | 2,000 x 6 | https://archive.ics.uci.edu/dataset/72/multiple+features | UCI CC BY 4.0 sentence | PASS | UCI donated not shown |
| 22 | 22 | mfeat-zernike | 2,000 x 47 | https://archive.ics.uci.edu/dataset/72/multiple+features | UCI CC BY 4.0 sentence | PASS | UCI donated not shown |
| 23 | 23 | cmc | 1,473 x 9 | https://archive.ics.uci.edu/dataset/30/contraceptive+method+choice | UCI CC BY 4.0 sentence | PASS | UCI donated 1997-07-06 |
| 28 | 28 | optdigits | 5,620 x 64 | https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits | UCI CC BY 4.0 sentence | PASS | UCI donated 1998-06-30 |
| 32 | 32 | pendigits | 10,992 x 16 | https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits | UCI CC BY 4.0 sentence | PASS | UCI donated 1998-06-30 |
| 182 | 2074 | satimage | 6,430 x 36 | https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite | UCI CC BY 4.0 sentence | PASS | UCI donated 1993-02-12 |
| 188 | 2079 | eucalyptus | 736 x 19 | https://ml.cms.waikato.ac.nz/weka/datasets.html | none found (page re-read 2026-11-08, no licence text) | UNVERIFIED |  |
| 307 | 3022 | vowel | 990 x 12 | https://archive.ics.uci.edu/dataset/152/connectionist+bench+vowel+recognition+deterding+data | UCI CC BY 4.0 sentence | PASS | UCI donated not shown |
| 300 | 3481 | isolet | 7,797 x 617 | https://archive.ics.uci.edu/dataset/54/isolet | UCI CC BY 4.0 sentence | PASS | UCI donated 1994-09-11 |
| 458 | 3549 | analcatdata_authorship | 841 x 70 | https://pages.stern.nyu.edu/~jsimonof/AnalCatData/ | none found (page re-read 2026-11-08, no licence text) | UNVERIFIED |  |
| 469 | 3560 | analcatdata_dmft | 797 x 4 | https://pages.stern.nyu.edu/~jsimonof/AnalCatData/ | none found (page re-read 2026-11-08, no licence text) | UNVERIFIED |  |
| 554 | 3573 | mnist_784 | 70,000 x 784 | http://yann.lecun.com/exdb/mnist/ | none found (page re-read 2026-11-08 (154 characters of text), no licence) | UNVERIFIED |  |
| 1497 | 9960 | wall-robot-navigation | 5,456 x 24 | https://archive.ics.uci.edu/dataset/194/wall+following+robot+navigation+data | UCI CC BY 4.0 sentence | PASS | UCI donated 2010-08-03 |
| 1501 | 9964 | semeion | 1,593 x 256 | https://archive.ics.uci.edu/dataset/178/semeion+handwritten+digit | UCI CC BY 4.0 sentence | PASS | UCI donated 2008-11-10 |
| 1468 | 9981 | cnae-9 | 1,080 x 856 | https://archive.ics.uci.edu/dataset/233/cnae+9 | UCI CC BY 4.0 sentence | PASS | UCI donated 2012-08-02 |
| 1475 | 9985 | first-order-theorem-proving | 6,118 x 51 | https://archive.ics.uci.edu/dataset/249/first+order+theorem+proving | UCI CC BY 4.0 sentence | PASS | UCI donated 2013-04-16 |
| 4538 | 14969 | GesturePhaseSegmentationProcessed | 9,873 x 32 | https://archive.ics.uci.edu/dataset/302/gesture+phase+segmentation | UCI CC BY 4.0 sentence | PASS | UCI donated 2014-06-17 |
| 1478 | 14970 | har | 10,299 x 561 | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones | UCI CC BY 4.0 sentence | PASS | UCI donated 2012-12-09 |
| 40499 | 125922 | texture | 5,500 x 40 | https://sci2s.ugr.es/keel/dataset.php?cod=72 | none found (KEEL page re-read 2026-11-08, no licence; ELENA origin page not reached) | UNVERIFIED |  |
| 40668 | 146195 | connect-4 | 67,557 x 42 | https://archive.ics.uci.edu/dataset/26/connect+4 | UCI CC BY 4.0 sentence | PASS | UCI donated 1995-02-03 |
| 40982 | 146817 | steel-plates-fault | 1,941 x 27 | https://archive.ics.uci.edu/dataset/198/steel+plates+faults | UCI CC BY 4.0 sentence | PASS | UCI donated 2010-10-25 |
| 40975 | 146821 | car | 1,728 x 6 | https://archive.ics.uci.edu/dataset/19/car+evaluation | UCI CC BY 4.0 sentence | PASS | UCI donated 1997-05-31 |
| 40984 | 146822 | segment | 2,310 x 19 | https://archive.ics.uci.edu/dataset/50/image+segmentation | UCI CC BY 4.0 sentence | PASS | UCI donated 1990-10-31 |
| 40979 | 146824 | mfeat-pixel | 2,000 x 240 | https://archive.ics.uci.edu/dataset/72/multiple+features | UCI CC BY 4.0 sentence | PASS | UCI donated not shown |
| 40996 | 146825 | Fashion-MNIST | 70,000 x 784 | https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE | "The MIT License (MIT) Copyright (c) 2017 Zalando SE ... to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software" | PASS |  |
| 41027 | 167119 | jungle_chess_2pcs_raw_endgame_complete | 44,819 x 6 | https://arxiv.org/abs/1604.07312 | none found (paper only; no licence for the generated data) | UNVERIFIED |  |
| 40923 | 167121 | Devnagari-Script | 92,000 x 1024 | https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset | UCI CC BY 4.0 sentence | PASS | UCI donated 2016-08-31 |
| 40927 | 167124 | CIFAR_10 | 60,000 x 3072 | https://www.cs.toronto.edu/~kriz/cifar.html | none found (page re-read 2026-11-08, no licence text) | UNVERIFIED |  |
