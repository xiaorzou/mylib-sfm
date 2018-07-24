import numpy as np
import pandas as pd
from scipy.linalg import svd


def saveData(outputfile, data, sfm):
    writer = pd.ExcelWriter(outputfile)
    data[sfm.getRegionName()].to_excel(writer, sheet_name=sfm.getRegionName())
    for SID in np.arange(len(sfm.getNumFactorSectors())):
        data[sfm.getSectorNames(SID)].to_excel(writer, sheet_name=sfm.getSectorNames(SID))

def normalizedInputData(data, sfm):
    data_PCA = {}
    corr_regionpart = data[sfm.getRegionName()].dot(data[sfm.getRegionName()].T)
    U, S = svd(corr_region_part)[0:2]
    num_factor_sectors = sfm.getNumFactorSectors()
    data_PCA[sfm.getRegionName()] = pd.DataFrame(data= U[:,0:sfm.getNumFactorRegion()].dot(np.diag(S[0:sfm.getNumFactorRegion()]**0.5)))
    for SID in np.arange(len(sfm.getNumFactorSectors())):
        corr_sector_part = data[sfm.getSectorNames(SID)].dot(data[sfm.getSectorNames(SID)].T)
        U, S = svd(corr_sector_part)[0:2]
        data_PCA[sfm.getSectorNames(SID)] = pd.DataFrame(data = U[:,0:num_factor_sectors[SID]].dot(np.diag(S[0:num_factor_sectors[SID]]**0.5)))
    return data_PCA