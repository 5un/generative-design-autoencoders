import numpy as np
import transforms3d

m1 = transforms3d.axangles.axangle2mat((0.0, 0.0, 1.0), np.radians(90))
m2 = transforms3d.axangles.axangle2mat((0.0, 1.0, 0.0), np.radians(90))

m3 = transforms3d.affines.compose(np.zeros(3), m1, np.ones(3))
m4 = transforms3d.affines.compose(np.zeros(3), m2, np.ones(3))

print(np.matmul(m3, m4))