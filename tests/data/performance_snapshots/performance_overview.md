# Performance Snapshot Overview

Generated: 2026-04-15T10:57:03+00:00

## `synthetic`
| model | variant | mean_dist | solver | device | knn | epoch_time(s) | gmm_fit_time(s) | inf_time(s) | updated_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| empirical | none |  | emd2 | cpu | 1.0000 | 7.2000 |  |  | 2026-04-15T10:46:09+00:00 |
| empirical | none |  | emd2 | gpu | 1.0000 | 7.7775 |  |  | 2026-04-15T10:51:50+00:00 |
| empirical | none |  | sinkhorn | cpu | 1.0000 | 0.8894 |  |  | 2026-04-15T10:52:41+00:00 |
| empirical | none |  | sinkhorn | gpu | 1.0000 | 0.0295 |  |  | 2026-04-15T10:52:57+00:00 |
| gmm | diag_bures | squared | emd2 | cpu | 1.0000 | 0.2971 | 2.7495 | 0.0006 | 2026-04-15T10:15:41+00:00 |
| gmm | diag_bures | squared | emd2 | gpu | 1.0000 | 0.3061 | 4.6895 | 0.0006 | 2026-04-07T15:50:28+00:00 |
| gmm | diag_bures | squared | sinkhorn | cpu | 1.0000 | 0.0382 | 2.9633 | 0.0006 | 2026-03-18T13:51:53+00:00 |
| gmm | diag_bures | squared | sinkhorn | gpu | 1.0000 | 0.0331 | 4.6895 | 0.0067 | 2026-04-07T15:50:45+00:00 |
| gmm | diag_bures | unsquared | emd2 | cpu | 1.0000 | 0.2964 | 2.7495 | 0.0006 | 2026-04-15T10:16:34+00:00 |
| gmm | diag_bures | unsquared | emd2 | gpu | 1.0000 | 0.2862 | 4.6895 | 0.0006 | 2026-04-07T15:51:18+00:00 |
| gmm | diag_bures | unsquared | sinkhorn | cpu | 1.0000 | 0.0409 | 2.7495 | 0.0111 | 2026-04-15T10:17:09+00:00 |
| gmm | diag_bures | unsquared | sinkhorn | gpu | 1.0000 | 0.0333 | 4.6895 | 0.0068 | 2026-04-07T15:51:35+00:00 |
| gmm | diag_bures+mi_reg | squared | emd2 | cpu | 1.0000 | 0.3203 | 2.7495 | 0.0006 | 2026-04-15T10:17:42+00:00 |
| gmm | diag_bures+mi_reg | squared | emd2 | gpu | 1.0000 | 0.2885 | 4.6895 | 0.0006 | 2026-04-07T15:52:11+00:00 |
| gmm | diag_bures+mi_reg | squared | sinkhorn | cpu | 1.0000 | 0.0439 | 2.9633 | 0.0006 | 2026-03-18T13:53:28+00:00 |
| gmm | diag_bures+mi_reg | squared | sinkhorn | gpu | 1.0000 | 0.0344 | 4.6895 | 0.0068 | 2026-04-07T15:52:28+00:00 |
| gmm | diag_bures+mi_reg | unsquared | emd2 | cpu | 1.0000 | 0.2860 | 2.7495 | 0.0006 | 2026-04-15T10:18:36+00:00 |
| gmm | diag_bures+mi_reg | unsquared | emd2 | gpu | 1.0000 | 0.2793 | 4.6895 | 0.0006 | 2026-04-07T15:53:02+00:00 |
| gmm | diag_bures+mi_reg | unsquared | sinkhorn | cpu | 1.0000 | 0.0249 | 2.7495 | 0.0068 | 2026-04-15T10:19:03+00:00 |
| gmm | diag_bures+mi_reg | unsquared | sinkhorn | gpu | 1.0000 | 0.0344 | 4.6895 | 0.0068 | 2026-04-07T15:53:19+00:00 |
| gmm | mi_reg | squared | emd2 | cpu | 1.0000 | 0.5670 | 2.7495 | 0.0009 | 2026-04-15T10:13:13+00:00 |
| gmm | mi_reg | squared | emd2 | gpu | 1.0000 | 0.6743 | 4.6895 | 0.0011 | 2026-04-07T15:48:06+00:00 |
| gmm | mi_reg | squared | sinkhorn | cpu | 1.0000 | 0.0714 | 2.9633 | 0.0012 | 2026-03-18T13:49:44+00:00 |
| gmm | mi_reg | squared | sinkhorn | gpu | 1.0000 | 0.0493 | 4.6895 | 0.0069 | 2026-04-07T15:48:25+00:00 |
| gmm | mi_reg | unsquared | emd2 | cpu | 1.0000 | 0.4954 | 2.7495 | 0.0009 | 2026-04-15T10:14:34+00:00 |
| gmm | mi_reg | unsquared | emd2 | gpu | 1.0000 | 0.5862 | 4.6895 | 0.0011 | 2026-04-07T15:49:32+00:00 |
| gmm | mi_reg | unsquared | sinkhorn | cpu | 1.0000 | 0.0375 | 2.7495 | 0.0070 | 2026-04-15T10:15:13+00:00 |
| gmm | mi_reg | unsquared | sinkhorn | gpu | 1.0000 | 0.0525 | 4.6895 | 0.0068 | 2026-04-07T15:49:51+00:00 |
| gmm | none | squared | emd2 | cpu | 1.0000 | 0.5101 | 2.7495 | 0.0009 | 2026-04-15T10:09:11+00:00 |
| gmm | none | squared | emd2 | gpu | 1.0000 | 1.5272 | 4.6895 | 0.0025 | 2026-04-07T15:45:10+00:00 |
| gmm | none | squared | sinkhorn | cpu | 1.0000 | 0.0847 | 2.9633 | 0.0013 | 2026-03-18T13:46:48+00:00 |
| gmm | none | squared | sinkhorn | gpu | 1.0000 | 0.0520 | 4.6895 | 0.0068 | 2026-04-07T15:45:32+00:00 |
| gmm | none | unsquared | emd2 | cpu | 1.0000 | 0.5481 | 2.7495 | 0.0009 | 2026-04-15T10:11:48+00:00 |
| gmm | none | unsquared | emd2 | gpu | 1.0000 | 0.4845 | 4.6895 | 0.0009 | 2026-04-07T15:46:32+00:00 |
| gmm | none | unsquared | sinkhorn | cpu | 1.0000 | 0.0347 | 2.7495 | 0.0073 | 2026-04-15T10:12:29+00:00 |
| gmm | none | unsquared | sinkhorn | gpu | 1.0000 | 0.0475 | 4.6895 | 0.0068 | 2026-04-07T15:46:50+00:00 |
| minibatch | none |  | emd2 | cpu | 0.8889 | 3.8995 |  |  | 2026-04-07T16:13:40+00:00 |

## `network`
| model | variant | mean_dist | solver | device | knn | epoch_time(s) | gmm_fit_time(s) | inf_time(s) | updated_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| empirical | none |  | emd2 | cpu | 0.8889 | 2.1022 |  |  | 2026-04-15T10:54:35+00:00 |
| empirical | none |  | emd2 | gpu | 0.8889 | 2.4005 |  |  | 2026-04-15T10:56:24+00:00 |
| empirical | none |  | sinkhorn | cpu | 0.8889 | 0.7633 |  |  | 2026-04-07T16:33:27+00:00 |
| empirical | none |  | sinkhorn | gpu | 0.8889 | 0.0331 |  |  | 2026-04-15T10:57:03+00:00 |
| gmm | diag_bures | squared | emd2 | cpu | 0.8889 | 0.3654 | 16.5701 | 0.0008 | 2026-04-15T10:29:23+00:00 |
| gmm | diag_bures | squared | emd2 | gpu | 0.8889 | 0.3839 | 9.8737 | 0.0010 | 2026-04-15T10:29:41+00:00 |
| gmm | diag_bures | squared | sinkhorn | cpu | 0.8889 | 0.0765 | 16.5701 | 0.0092 | 2026-04-15T10:30:25+00:00 |
| gmm | diag_bures | squared | sinkhorn | gpu | 0.8889 | 0.0851 | 9.8737 | 0.0103 | 2026-04-15T10:31:06+00:00 |
| gmm | diag_bures | unsquared | emd2 | cpu | 0.8889 | 0.4181 | 17.3930 | 0.0010 | 2026-04-07T16:06:21+00:00 |
| gmm | diag_bures | unsquared | emd2 | gpu | 0.8889 | 0.5063 | 9.8737 | 0.0012 | 2026-04-15T10:32:26+00:00 |
| gmm | diag_bures | unsquared | sinkhorn | cpu | 0.8889 | 0.0893 | 16.5701 | 0.0111 | 2026-04-15T10:32:50+00:00 |
| gmm | diag_bures | unsquared | sinkhorn | gpu | 0.8889 | 0.0817 | 15.7344 | 0.0009 | 2026-03-18T14:07:38+00:00 |
| gmm | diag_bures+mi_reg | squared | emd2 | cpu | 0.8889 | 0.3811 | 16.5701 | 0.0009 | 2026-04-15T10:33:23+00:00 |
| gmm | diag_bures+mi_reg | squared | emd2 | gpu | 0.8889 | 0.3692 | 9.8737 | 0.0009 | 2026-04-15T10:33:36+00:00 |
| gmm | diag_bures+mi_reg | squared | sinkhorn | cpu | 0.8889 | 0.0584 | 16.5701 | 0.0068 | 2026-04-15T10:33:49+00:00 |
| gmm | diag_bures+mi_reg | squared | sinkhorn | gpu | 0.8889 | 0.0622 | 9.8737 | 0.0068 | 2026-04-15T10:34:02+00:00 |
| gmm | diag_bures+mi_reg | unsquared | emd2 | cpu | 0.8889 | 0.4324 | 24.6853 | 0.0008 | 2026-03-18T14:09:18+00:00 |
| gmm | diag_bures+mi_reg | unsquared | emd2 | gpu | 0.8889 | 0.3416 | 9.7599 | 0.0008 | 2026-04-07T16:10:48+00:00 |
| gmm | diag_bures+mi_reg | unsquared | sinkhorn | cpu | 0.8889 | 0.0677 | 24.6853 | 0.0010 | 2026-03-18T14:10:18+00:00 |
| gmm | diag_bures+mi_reg | unsquared | sinkhorn | gpu | 0.8889 | 0.1078 | 15.7344 | 0.0013 | 2026-03-18T14:10:33+00:00 |
| gmm | mi_reg | squared | emd2 | cpu | 0.8889 | 0.8519 | 16.5701 | 0.0017 | 2026-04-15T10:25:13+00:00 |
| gmm | mi_reg | squared | emd2 | gpu | 0.8889 | 0.7356 | 9.8737 | 0.0015 | 2026-04-15T10:25:34+00:00 |
| gmm | mi_reg | squared | sinkhorn | cpu | 0.8889 | 0.1369 | 16.5701 | 0.0075 | 2026-04-15T10:25:49+00:00 |
| gmm | mi_reg | squared | sinkhorn | gpu | 0.8889 | 0.1739 | 9.8737 | 0.0098 | 2026-04-15T10:26:09+00:00 |
| gmm | mi_reg | unsquared | emd2 | cpu | 0.8889 | 0.7527 | 17.3930 | 0.0017 | 2026-04-07T16:02:18+00:00 |
| gmm | mi_reg | unsquared | emd2 | gpu | 0.8889 | 0.6600 | 9.8737 | 0.0014 | 2026-04-15T10:28:29+00:00 |
| gmm | mi_reg | unsquared | sinkhorn | cpu | 0.8333 | 0.1317 | 16.5701 | 0.0074 | 2026-04-15T10:28:49+00:00 |
| gmm | mi_reg | unsquared | sinkhorn | gpu | 0.8889 | 0.1410 | 9.7599 | 0.0098 | 2026-04-07T16:04:12+00:00 |
| gmm | none | squared | emd2 | cpu | 0.8889 | 0.6267 | 16.5701 | 0.0013 | 2026-04-15T10:20:28+00:00 |
| gmm | none | squared | emd2 | gpu | 0.8333 | 1.1792 | 9.7599 | 0.0022 | 2026-04-07T15:55:08+00:00 |
| gmm | none | squared | sinkhorn | cpu | 0.8889 | 0.2191 | 16.5701 | 0.0093 | 2026-04-15T10:21:33+00:00 |
| gmm | none | squared | sinkhorn | gpu | 0.8889 | 0.1345 | 9.8737 | 0.0081 | 2026-04-15T10:22:10+00:00 |
| gmm | none | unsquared | emd2 | cpu | 0.8889 | 0.6213 | 17.3930 | 0.0014 | 2026-04-07T15:57:51+00:00 |
| gmm | none | unsquared | emd2 | gpu | 0.8889 | 0.6339 | 9.8737 | 0.0014 | 2026-04-15T10:24:08+00:00 |
| gmm | none | unsquared | sinkhorn | cpu | 0.8889 | 0.1436 | 24.6853 | 0.0013 | 2026-03-18T13:58:38+00:00 |
| gmm | none | unsquared | sinkhorn | gpu | 0.8333 | 0.1492 | 9.8737 | 0.0083 | 2026-04-15T10:24:52+00:00 |
| minibatch | none |  | emd2 | cpu | 0.8333 | 1.2897 |  |  | 2026-04-15T10:40:59+00:00 |
