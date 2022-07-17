import pandas as pd
import numpy as np
import random

s = r"""
python "c:\Users\PC\Documents\tony\Ternary-Metal-Nitrides-Research\src/cgcnn-mod-transformers/main.py" --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 20 "c:\Users\PC\Documents\tony\Ternary-Metal-Nitrides-Research\src/data"
OUTPUT >> Epoch: [0][0/18]	Time 6.750 (6.750)	Data 5.749 (5.749)	Loss 3.3729 (3.3729)	MAE 117.867 (117.867)
Epoch: [0][10/18]	Time 4.590 (5.257)	Data 3.603 (4.058)	Loss 0.8073 (3.0405)	MAE 48.669 (94.539)
Test: [0/6]	Time 4.679 (4.679)	Loss 0.8847 (0.8847)	MAE 57.510 (57.510)
 * MAE 56.231
Epoch: [1][0/18]	Time 0.972 (0.972)	Data 0.028 (0.028)	Loss 0.8769 (0.8769)	MAE 56.798 (56.798)
Epoch: [1][10/18]	Time 1.508 (1.101)	Data 0.021 (0.015)	Loss 0.5600 (0.7736)	MAE 44.885 (51.831)
Test: [0/6]	Time 0.250 (0.250)	Loss 0.6272 (0.6272)	MAE 37.322 (37.322)
 * MAE 32.684
Epoch: [2][0/18]	Time 1.069 (1.069)	Data 0.021 (0.021)	Loss 0.5012 (0.5012)	MAE 30.793 (30.793)
Epoch: [2][10/18]	Time 1.023 (1.170)	Data 0.014 (0.015)	Loss 0.2106 (0.2784)	MAE 23.562 (26.393)
Test: [0/6]	Time 0.277 (0.277)	Loss 0.2270 (0.2270)	MAE 19.917 (19.917)
 * MAE 20.963
Epoch: [3][0/18]	Time 1.039 (1.039)	Data 0.019 (0.019)	Loss 0.1282 (0.1282)	MAE 18.342 (18.342)
Epoch: [3][10/18]	Time 1.094 (1.074)	Data 0.013 (0.014)	Loss 0.1397 (0.1906)	MAE 18.174 (19.785)
Test: [0/6]	Time 0.275 (0.275)	Loss 0.1646 (0.1646)	MAE 22.374 (22.374)
 * MAE 24.323
Epoch: [4][0/18]	Time 0.994 (0.994)	Data 0.020 (0.020)	Loss 0.1196 (0.1196)	MAE 18.796 (18.796)
Epoch: [4][10/18]	Time 0.956 (0.940)	Data 0.011 (0.012)	Loss 0.2787 (0.1936)	MAE 22.867 (20.017)
Test: [0/6]	Time 0.250 (0.250)	Loss 0.2055 (0.2055)	MAE 24.861 (24.861)
 * MAE 24.944
Epoch: [5][0/18]	Time 0.919 (0.919)	Data 0.016 (0.016)	Loss 0.2905 (0.2905)	MAE 28.985 (28.985)
Epoch: [5][10/18]	Time 0.951 (0.942)	Data 0.011 (0.011)	Loss 0.1304 (0.1972)	MAE 18.853 (20.666)
Test: [0/6]	Time 0.286 (0.286)	Loss 0.2535 (0.2535)	MAE 18.939 (18.939)
 * MAE 19.325
Epoch: [6][0/18]	Time 0.902 (0.902)	Data 0.017 (0.017)	Loss 0.1282 (0.1282)	MAE 17.604 (17.604)
Epoch: [6][10/18]	Time 0.993 (0.957)	Data 0.014 (0.013)	Loss 0.1530 (0.1850)	MAE 16.774 (18.120)
Test: [0/6]	Time 0.249 (0.249)	Loss 0.0928 (0.0928)	MAE 15.249 (15.249)
 * MAE 16.517
Epoch: [7][0/18]	Time 1.031 (1.031)	Data 0.022 (0.022)	Loss 0.1257 (0.1257)	MAE 17.711 (17.711)
Epoch: [7][10/18]	Time 0.882 (0.966)	Data 0.009 (0.013)	Loss 0.2425 (0.1681)	MAE 16.617 (17.349)
Test: [0/6]	Time 0.366 (0.366)	Loss 0.1285 (0.1285)	MAE 15.294 (15.294)
 * MAE 16.529
Epoch: [8][0/18]	Time 1.840 (1.840)	Data 0.041 (0.041)	Loss 0.1251 (0.1251)	MAE 18.055 (18.055)
Epoch: [8][10/18]	Time 1.794 (1.407)	Data 0.017 (0.020)	Loss 0.1414 (0.1424)	MAE 15.544 (16.667)
Test: [0/6]	Time 0.444 (0.444)	Loss 0.1622 (0.1622)	MAE 20.074 (20.074)
 * MAE 19.632
Epoch: [9][0/18]	Time 1.256 (1.256)	Data 0.092 (0.092)	Loss 0.1535 (0.1535)	MAE 17.575 (17.575)
Epoch: [9][10/18]	Time 0.862 (1.102)	Data 0.012 (0.021)	Loss 0.0887 (0.1374)	MAE 14.210 (16.411)
Test: [0/6]	Time 0.311 (0.311)	Loss 0.1052 (0.1052)	MAE 15.266 (15.266)
 * MAE 15.563
Epoch: [10][0/18]	Time 1.188 (1.188)	Data 0.024 (0.024)	Loss 0.1043 (0.1043)	MAE 15.483 (15.483)
Epoch: [10][10/18]	Time 1.007 (1.093)	Data 0.014 (0.015)	Loss 0.1042 (0.1508)	MAE 16.080 (15.628)
Test: [0/6]	Time 0.301 (0.301)	Loss 0.0892 (0.0892)	MAE 14.077 (14.077)
 * MAE 15.416
Epoch: [11][0/18]	Time 1.036 (1.036)	Data 0.020 (0.020)	Loss 0.0840 (0.0840)	MAE 13.376 (13.376)
Epoch: [11][10/18]	Time 1.003 (1.030)	Data 0.012 (0.014)	Loss 0.1025 (0.1749)	MAE 15.227 (17.762)
Test: [0/6]	Time 0.300 (0.300)	Loss 0.2122 (0.2122)	MAE 17.369 (17.369)
 * MAE 17.602
Epoch: [12][0/18]	Time 1.310 (1.310)	Data 0.020 (0.020)	Loss 0.2841 (0.2841)	MAE 21.847 (21.847)
Epoch: [12][10/18]	Time 1.105 (1.079)	Data 0.011 (0.014)	Loss 0.1135 (0.1511)	MAE 13.862 (16.336)
Test: [0/6]	Time 0.270 (0.270)	Loss 0.2253 (0.2253)	MAE 17.697 (17.697)
 * MAE 15.844
Epoch: [13][0/18]	Time 1.147 (1.147)	Data 0.021 (0.021)	Loss 0.1136 (0.1136)	MAE 16.702 (16.702)
Epoch: [13][10/18]	Time 0.996 (0.985)	Data 0.011 (0.014)	Loss 0.2497 (0.1490)	MAE 21.087 (16.585)
Test: [0/6]	Time 0.377 (0.377)	Loss 0.3423 (0.3423)	MAE 18.583 (18.583)
 * MAE 16.528
Epoch: [14][0/18]	Time 1.032 (1.032)	Data 0.019 (0.019)	Loss 0.0928 (0.0928)	MAE 14.844 (14.844)
Epoch: [14][10/18]	Time 0.933 (0.978)	Data 0.011 (0.013)	Loss 0.2823 (0.1568)	MAE 17.936 (16.345)
Test: [0/6]	Time 0.281 (0.281)	Loss 0.3458 (0.3458)	MAE 18.546 (18.546)
 * MAE 16.153
Epoch: [15][0/18]	Time 1.020 (1.020)	Data 0.019 (0.019)	Loss 0.0997 (0.0997)	MAE 16.260 (16.260)
Epoch: [15][10/18]	Time 1.606 (1.384)	Data 0.016 (0.016)	Loss 0.1302 (0.1114)	MAE 17.321 (14.910)
Test: [0/6]	Time 0.305 (0.305)	Loss 0.0997 (0.0997)	MAE 14.331 (14.331)
 * MAE 15.971
Epoch: [16][0/18]	Time 1.014 (1.014)	Data 0.015 (0.015)	Loss 0.2573 (0.2573)	MAE 18.694 (18.694)
Epoch: [16][10/18]	Time 1.009 (1.183)	Data 0.016 (0.017)	Loss 0.0916 (0.1249)	MAE 14.612 (16.167)
Test: [0/6]	Time 0.281 (0.281)	Loss 0.2870 (0.2870)	MAE 19.111 (19.111)
 * MAE 17.582
Epoch: [17][0/18]	Time 0.935 (0.935)	Data 0.021 (0.021)	Loss 0.1900 (0.1900)	MAE 14.851 (14.851)
Epoch: [17][10/18]	Time 1.513 (1.212)	Data 0.032 (0.017)	Loss 0.0944 (0.1505)	MAE 14.321 (15.815)
Test: [0/6]	Time 0.517 (0.517)	Loss 0.1404 (0.1404)	MAE 16.707 (16.707)
 * MAE 15.974
Epoch: [18][0/18]	Time 1.510 (1.510)	Data 0.027 (0.027)	Loss 0.0690 (0.0690)	MAE 12.808 (12.808)
Epoch: [18][10/18]	Time 0.971 (1.208)	Data 0.014 (0.016)	Loss 0.0743 (0.1094)	MAE 12.890 (14.767)
Test: [0/6]	Time 0.294 (0.294)	Loss 0.1243 (0.1243)	MAE 17.897 (17.897)
 * MAE 17.715
Epoch: [19][0/18]	Time 1.291 (1.291)	Data 0.031 (0.031)	Loss 0.1270 (0.1270)	MAE 18.162 (18.162)
Epoch: [19][10/18]	Time 0.942 (1.077)	Data 0.012 (0.015)	Loss 0.0793 (0.1327)	MAE 14.130 (15.127)
Test: [0/6]	Time 0.239 (0.239)	Loss 0.2241 (0.2241)	MAE 17.985 (17.985)
 * MAE 16.901
---------Evaluate Model on Test Set---------------
Test: [0/6]	Time 3.048 (3.048)	Loss 0.2564 (0.2564)	MAE 17.723 (17.723)
 ** MAE 16.762

ERROR >> c:\Users\PC\Documents\tony\Ternary-Metal-Nitrides-Research\venv\lib\site-packages\pymatgen\io\cif.py:1160: UserWarning: Issues encountered while parsing CIF: Some fractional coordinates rounded to ideal values to avoid issues with finite precision.
  warnings.warn("Issues encountered while parsing CIF: %s" % "\n".join(self.warnings))

"""

for i in s.splitlines():
    if 'MAE' in i:
        print(f" >> {i}")