'''
This package implement the algorithm described in the following paper:
Zou, Xiaorong
"Structured Factor Model and its Applications on Market Risk Management"
the paper can be found in:

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3180413


inputs:
1)	signature: is used to describe input file name.	'N50_m2_s23222.xlsx' has the following meaning:
        N=50 (number variables), number of global factor m=2, there are 5 sectors with number of factors [2, 3, 2, 2, 2].
2)	boundary: describe the boundary position for each sector. For instance, if boundary = [0, 10, 20, 30, 40, 50],
    sector 0 refers to positions from 0 to 9, sector 1 refers to position from 10 to 19 and so on.
2) the file 'model' contains model values of factor loadings for global factors, sector factors and specific factors.
In real application, only emprical correlation matrix is available. this is only for testing purpose.
3)	the file 'init' contains the initial guess values of factor loadings used in calibration process.	This file is not necessary,
4)	the variable 'upperbound_constrain. define the square of upper bound cut-off for global and sector factor loadings.
outputs: output are saved to two files
1) 'calibration' contains
a)	the calibrated factor loadings for global factors, sector factors and specific risk factors
b)	the validation values, i.e. the difference of the model correlations and the calibrated correlations
2) 'target_PCA_N50_r2_s23222.xlsx' contains the PCA normalized factor loadings based on model values.
remark: If the model works fine, then
1)	the difference of the model correlations and the calibrated correlations should be small
2)	the calibrated factor loadings (in calibration) should be close the normalized factor loadings (in target_PCA)
'''


'''
how to run:  the code can be run without any input.
after the first run,  you can comment out line 56.  so your results will not depend on the initial guess in optimization.
'''

import numpy as np
import pandas as pd
from sfm import sfm
from util_sfm import normalizedInputData
from util_sfm import saveData
import unittest

class MyTestCase(unittest.TestCase):
    def test1(self):
        signature = 'N50_m2_s23222.xlsx'
        inputfile       = 'model_' + signature
        initdatafile = 'init_' + signature
        outputfile = 'calibration_' + signature
        outputfile_PCA = 'target_PCA_' + signature

        num_factor_region = 2
        boundary =  [0, 10, 20, 30,40,50]
        num_factor_sectors = [2,3, 2, 2, 2]
        upperbound_constrain = 0.95**2

        xls = pd.ExcelFile(inputfile)
        all_beta = np.matrix(xls.parse('beta_full'))
        corrMatrix = all_beta.dot(all_beta.T)
        corrMatrix = corrMatrix - np.diag(np.diag(corrMatrix)) + np.identity(boundary[-1])
        sfm_obj = sfm(num_factor_region, num_factor_sectors, boundary, corrMatrix, upperbound_constrain, initdatafile)

        #to use a randomly generated initial data for calibration, one should call the following.
        # Once a init file is generated, comment the line out and rerun the program.
        sfm_obj.generateInitData(initdatafile)

        sfm_obj.loadInitData()
        initdata = sfm_obj.getInitData()
        sfm_obj.calibration()
        data = {}
        data[sfm_obj.getRegionName()] = xls.parse(sfm_obj.getRegionName())
        for SID in np.arange(len(sfm_obj.getNumFactorSectors())):
            data[sfm_obj.getSectorNames(SID)] = xls.parse(sfm_obj.getSectorNames(SID))

        data_PCA = normalizedInputData(data, sfm_obj)
        saveData(outputfile_PCA, data_PCA, sfm_obj)
        sfm_obj.saveCalibrationOutput(outputfile)


if __name__ == '__main__':
    unittest.main()