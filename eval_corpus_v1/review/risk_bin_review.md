# Risk bin review checklist

Draft bins were computed from `impact_level × structure_weight` (see README).
For each row, set **Approved bin** (or write `OK` to accept the draft), then we will
copy values into `risk_bin_gold` in `incidents.csv`.

| ID | Stratum | Suppliers | g_reach | Impact | Draft bin | Score | Rationale | Approved bin |
|---|---|---|---:|---|---|---:|---|---|
| INC_001 | bottleneck_sole | H-ONE Co., Ltd. | 56 | severe | **very_high** | 6.00 | impact=severe(w=2.0) × structure(g=56,total=108,crit=False,w=3.0) → score=6.00 | |
| INC_002 | bottleneck_sole | HI-LEX Corporation | 49 | severe | **high** | 4.00 | impact=severe(w=2.0) × structure(g=49,total=114,crit=False,w=2.0) → score=4.00 | |
| INC_003 | bottleneck_sole | Saga Tekkohsho Co., Ltd. | 49 | severe | **high** | 4.00 | impact=severe(w=2.0) × structure(g=49,total=51,crit=False,w=2.0) → score=4.00 | |
| INC_004 | bottleneck_sole | Minebea AccessSolutions Thai Ltd. (Formerly Honda Lock Th... | 40 | severe | **high** | 4.00 | impact=severe(w=2.0) × structure(g=40,total=172,crit=False,w=2.0) → score=4.00 | |
| INC_005 | bottleneck_sole | Pacific Engineering Corporation | 40 | severe | **high** | 4.00 | impact=severe(w=2.0) × structure(g=40,total=43,crit=False,w=2.0) → score=4.00 | |
| INC_006 | bottleneck_sole | Mitsuba Corporation | 35 | severe | **high** | 4.00 | impact=severe(w=2.0) × structure(g=35,total=272,crit=False,w=2.0) → score=4.00 | |
| INC_007 | bottleneck_sole | Bestex (Thailand) Co., Ltd. | 35 | severe | **high** | 4.00 | impact=severe(w=2.0) × structure(g=35,total=93,crit=False,w=2.0) → score=4.00 | |
| INC_008 | bottleneck_sole | MAHLE Behr Thermal Systems (Thailand) Co., Ltd. (Formerly... | 34 | severe | **high** | 4.00 | impact=severe(w=2.0) × structure(g=34,total=82,crit=False,w=2.0) → score=4.00 | |
| INC_009 | bottleneck_sole | Yutaka Giken Co., Ltd. | 32 | severe | **high** | 4.00 | impact=severe(w=2.0) × structure(g=32,total=161,crit=False,w=2.0) → score=4.00 | |
| INC_010 | bottleneck_sole | Astemo, Ltd. (Formerly Hitachi Astemo, Ltd. ) | 125 | severe | **very_high** | 6.00 | impact=severe(w=2.0) × structure(g=125,total=452,crit=True,w=3.0) → score=6.00 | |
| INC_011 | bottleneck_sole | Denso Corporation | 80 | minor | **medium** | 3.00 | impact=minor(w=1.0) × structure(g=80,total=413,crit=True,w=3.0) → score=3.00 | |
| INC_012 | bottleneck_sole | Foshan Fengfu Auto Parts Co., Ltd. | 58 | minor | **medium** | 3.00 | impact=minor(w=1.0) × structure(g=58,total=204,crit=True,w=3.0) → score=3.00 | |
| INC_013 | bottleneck_sole | Mitsubishi Electric Corporation | 52 | minor | **medium** | 3.00 | impact=minor(w=1.0) × structure(g=52,total=288,crit=True,w=3.0) → score=3.00 | |
| INC_014 | bottleneck_sole | Nissin Manufacturing Group Co., Ltd. (Formerly Nissin Man... | 48 | none | **very_low** | 0.00 | impact=none(w=0.0) × structure(g=48,total=59,crit=True,w=3.0) → score=0.00 | |
| INC_015 | critical_part_sole | Hitachi Astemo Chassis Systems (Zhongshan) Ltd. (Formerly... | 8 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=8,total=36,crit=True,w=0.5) → score=1.00 | |
| INC_016 | critical_part_sole | Joyson Safety Systems (Jingzhou) Co., Ltd. (Formerly Taka... | 6 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=6,total=87,crit=True,w=0.5) → score=1.00 | |
| INC_017 | critical_part_sole | Dongfeng Honda Auto Parts Co., Ltd. | 28 | severe | **medium** | 2.00 | impact=severe(w=2.0) × structure(g=28,total=103,crit=True,w=1.0) → score=2.00 | |
| INC_018 | critical_part_sole | Autoliv (Guangzhou) Vehicle Safety Systems Co., Ltd. | 8 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=8,total=214,crit=True,w=0.5) → score=1.00 | |
| INC_019 | critical_part_sole | Changzhou Gohsyu Automobile Parts Co., Ltd.(CGA) | 16 | severe | **medium** | 2.00 | impact=severe(w=2.0) × structure(g=16,total=16,crit=True,w=1.0) → score=2.00 | |
| INC_020 | critical_part_sole | Dongfeng Honda Engine Co., Ltd. | 13 | severe | **medium** | 2.00 | impact=severe(w=2.0) × structure(g=13,total=43,crit=True,w=1.0) → score=2.00 | |
| INC_021 | critical_part_sole | Bosch Automotive Products (Suzhou) Co., Ltd. | 9 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=9,total=155,crit=True,w=0.5) → score=1.00 | |
| INC_022 | critical_part_sole | Nidec Elesys (Zhongshan) Corporation | 14 | severe | **medium** | 2.00 | impact=severe(w=2.0) × structure(g=14,total=107,crit=True,w=1.0) → score=2.00 | |
| INC_023 | critical_part_sole | Honda Automobile (Thailand), Ayutthaya Plant | 17 | minor | **low** | 1.00 | impact=minor(w=1.0) × structure(g=17,total=130,crit=True,w=1.0) → score=1.00 | |
| INC_024 | critical_part_sole | Continental Automotive Systems (Shanghai) Co., Ltd. | 13 | minor | **low** | 1.00 | impact=minor(w=1.0) × structure(g=13,total=51,crit=True,w=1.0) → score=1.00 | |
| INC_025 | critical_part_sole | Huizhou Zhurun Automotive Wire Co., Ltd. | 10 | minor | **low** | 1.00 | impact=minor(w=1.0) × structure(g=10,total=10,crit=True,w=1.0) → score=1.00 | |
| INC_026 | critical_part_sole | Bosch (Robert Bosch LLC) | 14 | none | **very_low** | 0.00 | impact=none(w=0.0) × structure(g=14,total=255,crit=True,w=1.0) → score=0.00 | |
| INC_027 | redundant_high_reach | Stanley Electric U.S. Co., Inc. | 0 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=0,total=360,crit=False,w=0.5) → score=1.00 | |
| INC_028 | redundant_high_reach | Lumax Industries Ltd. | 0 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=0,total=237,crit=False,w=0.5) → score=1.00 | |
| INC_029 | redundant_high_reach | The Yokohama Rubber Co., Ltd. | 0 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=0,total=226,crit=False,w=0.5) → score=1.00 | |
| INC_030 | redundant_high_reach | Bridgestone Corporation | 0 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=0,total=226,crit=False,w=0.5) → score=1.00 | |
| INC_031 | redundant_high_reach | Bridgestone Americas, Inc. | 0 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=0,total=223,crit=False,w=0.5) → score=1.00 | |
| INC_032 | redundant_high_reach | SMR Automotive Systems India Limited - Noida | 0 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=0,total=222,crit=False,w=0.5) → score=1.00 | |
| INC_033 | redundant_high_reach | Hangzhou Yokohama Tire Co., Ltd. | 0 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=0,total=222,crit=False,w=0.5) → score=1.00 | |
| INC_034 | redundant_high_reach | Shanghai Michelin Tire Co., Ltd. (Formerly Shanghai Miche... | 0 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=0,total=222,crit=False,w=0.5) → score=1.00 | |
| INC_035 | redundant_high_reach | Murakami Corporation | 0 | minor | **very_low** | 0.50 | impact=minor(w=1.0) × structure(g=0,total=222,crit=False,w=0.5) → score=0.50 | |
| INC_036 | redundant_high_reach | Industrias Michelin, S.A. De C.V. - León Mx2 | 0 | minor | **very_low** | 0.50 | impact=minor(w=1.0) × structure(g=0,total=220,crit=False,w=0.5) → score=0.50 | |
| INC_037 | redundant_high_reach | JK Tyre & Industries Ltd. | 0 | minor | **very_low** | 0.50 | impact=minor(w=1.0) × structure(g=0,total=220,crit=False,w=0.5) → score=0.50 | |
| INC_038 | redundant_high_reach | Xiamen Zhengxin Rubber Industry Co., Ltd. | 0 | none | **very_low** | 0.00 | impact=none(w=0.0) × structure(g=0,total=220,crit=False,w=0.5) → score=0.00 | |
| INC_039 | moderate_reach | Toyo Denso Co., Ltd. | 29 | severe | **medium** | 2.00 | impact=severe(w=2.0) × structure(g=29,total=127,crit=False,w=1.0) → score=2.00 | |
| INC_040 | moderate_reach | Fukoku Co., Ltd. | 29 | severe | **medium** | 2.00 | impact=severe(w=2.0) × structure(g=29,total=51,crit=False,w=1.0) → score=2.00 | |
| INC_041 | moderate_reach | G-TEKT (Thailand) Co.,Ltd. | 29 | severe | **medium** | 2.00 | impact=severe(w=2.0) × structure(g=29,total=29,crit=False,w=1.0) → score=2.00 | |
| INC_042 | moderate_reach | Nifco Inc. | 27 | severe | **medium** | 2.00 | impact=severe(w=2.0) × structure(g=27,total=66,crit=True,w=1.0) → score=2.00 | |
| INC_043 | moderate_reach | F.C.C. Co., Ltd. | 27 | severe | **medium** | 2.00 | impact=severe(w=2.0) × structure(g=27,total=58,crit=True,w=1.0) → score=2.00 | |
| INC_044 | moderate_reach | Yorozu Corp. | 27 | severe | **medium** | 2.00 | impact=severe(w=2.0) × structure(g=27,total=58,crit=False,w=1.0) → score=2.00 | |
| INC_045 | moderate_reach | NGK Ceramics Suzhou Co., Ltd. | 27 | severe | **medium** | 2.00 | impact=severe(w=2.0) × structure(g=27,total=30,crit=False,w=1.0) → score=2.00 | |
| INC_046 | moderate_reach | Nihon Plast Co., Ltd. | 26 | minor | **low** | 1.00 | impact=minor(w=1.0) × structure(g=26,total=169,crit=True,w=1.0) → score=1.00 | |
| INC_047 | moderate_reach | THK Rhythm (Thailand) Co., Ltd. | 26 | minor | **low** | 1.00 | impact=minor(w=1.0) × structure(g=26,total=26,crit=False,w=1.0) → score=1.00 | |
| INC_048 | moderate_reach | Honda | 25 | minor | **low** | 1.00 | impact=minor(w=1.0) × structure(g=25,total=168,crit=True,w=1.0) → score=1.00 | |
| INC_049 | moderate_reach | Denso International America, Inc. | 24 | minor | **low** | 1.00 | impact=minor(w=1.0) × structure(g=24,total=264,crit=False,w=1.0) → score=1.00 | |
| INC_050 | moderate_reach | Thai Steel Cable Public Co., Ltd. | 24 | none | **very_low** | 0.00 | impact=none(w=0.0) × structure(g=24,total=109,crit=False,w=1.0) → score=0.00 | |
| INC_051 | low_reach | Magna International Inc. | 1 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=1,total=223,crit=False,w=0.5) → score=1.00 | |
| INC_052 | low_reach | Minebea AccessSolutions Inc. (Formerly Honda Lock Mfg. Co... | 1 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=1,total=223,crit=False,w=0.5) → score=1.00 | |
| INC_053 | low_reach | American Mitsuba Indiana - AMI | 1 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=1,total=198,crit=False,w=0.5) → score=1.00 | |
| INC_054 | low_reach | Bosch (Robert Bosch GmbH) | 1 | severe | **low** | 1.00 | impact=severe(w=2.0) × structure(g=1,total=192,crit=False,w=0.5) → score=1.00 | |
| INC_055 | low_reach | Pittsburgh Glass Works, LLC. | 1 | minor | **very_low** | 0.50 | impact=minor(w=1.0) × structure(g=1,total=175,crit=False,w=0.5) → score=0.50 | |
| INC_056 | low_reach | Samvardhana Motherson Group | 1 | minor | **very_low** | 0.50 | impact=minor(w=1.0) × structure(g=1,total=145,crit=False,w=0.5) → score=0.50 | |
| INC_057 | low_reach | Wuhan Sumiden Wiring Systems Co., Ltd. | 1 | minor | **very_low** | 0.50 | impact=minor(w=1.0) × structure(g=1,total=144,crit=True,w=0.5) → score=0.50 | |
| INC_058 | low_reach | Cardington Yutaka Technologies, Inc. | 1 | minor | **very_low** | 0.50 | impact=minor(w=1.0) × structure(g=1,total=125,crit=False,w=0.5) → score=0.50 | |
| INC_059 | low_reach | MAHLE Behr USA Inc. | 1 | none | **very_low** | 0.00 | impact=none(w=0.0) × structure(g=1,total=125,crit=False,w=0.5) → score=0.00 | |
| INC_060 | low_reach | HI-LEX America Inc. | 1 | none | **very_low** | 0.00 | impact=none(w=0.0) × structure(g=1,total=105,crit=False,w=0.5) → score=0.00 | |
| INC_061 | multi_supplier | Astemo, Ltd. (Formerly Hitachi Astemo, Ltd. ); Stanley El... | 125 | severe | **very_high** | 6.00 | impact=severe(w=2.0) × structure(g=125,total=452,crit=True,w=3.0) → score=6.00 | |
| INC_062 | multi_supplier | Denso Corporation; Lumax Industries Ltd. | 80 | severe | **very_high** | 6.00 | impact=severe(w=2.0) × structure(g=80,total=413,crit=True,w=3.0) → score=6.00 | |
| INC_063 | multi_supplier | Foshan Fengfu Auto Parts Co., Ltd.; The Yokohama Rubber C... | 58 | severe | **very_high** | 6.00 | impact=severe(w=2.0) × structure(g=58,total=226,crit=True,w=3.0) → score=6.00 | |
| INC_064 | multi_supplier | H-ONE Co., Ltd.; Bridgestone Corporation | 56 | severe | **very_high** | 6.00 | impact=severe(w=2.0) × structure(g=56,total=226,crit=False,w=3.0) → score=6.00 | |
| INC_065 | multi_supplier | Nissin Manufacturing Group Co., Ltd. (Formerly Nissin Man... | 48 | severe | **very_high** | 6.00 | impact=severe(w=2.0) × structure(g=48,total=59,crit=True,w=3.0) → score=6.00 | |
| INC_066 | multi_supplier | Hitachi Astemo Americas, Inc. (Formerly Hitachi Automotiv... | 43 | severe | **very_high** | 6.00 | impact=severe(w=2.0) × structure(g=43,total=319,crit=True,w=3.0) → score=6.00 | |
| INC_067 | multi_supplier | Hitachi Astemo Automotive Powertrain Technologies (Donggu... | 43 | severe | **very_high** | 6.00 | impact=severe(w=2.0) × structure(g=43,total=220,crit=True,w=3.0) → score=6.00 | |
| INC_068 | multi_supplier | Industrias Michelin, S.A. De C.V. - León Mx2; Honda; Paci... | 40 | minor | **medium** | 3.00 | impact=minor(w=1.0) × structure(g=40,total=220,crit=True,w=3.0) → score=3.00 | |
| INC_069 | multi_supplier | JK Tyre & Industries Ltd.; Denso International America, I... | 37 | minor | **medium** | 3.00 | impact=minor(w=1.0) × structure(g=37,total=264,crit=True,w=3.0) → score=3.00 | |
| INC_070 | multi_supplier | Xiamen Zhengxin Rubber Industry Co., Ltd.; Thai Steel Cab... | 35 | none | **very_low** | 0.00 | impact=none(w=0.0) × structure(g=35,total=272,crit=False,w=2.0) → score=0.00 | |

## How to approve

1. Fill the Approved bin column with one of: `very_low`, `low`, `medium`, `high`, `very_high`, or `OK`.
2. Tell the agent to freeze gold from this review (or paste overrides).
3. Re-run `python validate.py` after `risk_bin_gold` is populated.
