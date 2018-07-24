'''
This classes implement the algorithm described in the following paper:
Zou, Xiaorong:  "Structured Factor Model and its Applications on Market Risk Management"

the paper can be found in:   <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3180413>
'''

import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.optimize import minimize

class sfm( object ):
    def __init__(self, num_factor_region, num_factor_sectors, boundary, corrMatrix, upperbound_constrain, initdatafile):
        self._corrMatrix = corrMatrix
        self._num_factor_region = num_factor_region
        self._num_factor_sectors = num_factor_sectors
        self._boundary = boundary
        self._upperbound_constrain = upperbound_constrain
        self._initdata = {}
        self._initdatafile = initdatafile
        self._shape='shape'
        self._weight= 'weight'
        self._corr='corr'
        self._selection= 'selection'
        self._beta_region= 'beta_region'
        self._beta_region_validation= 'beta_region_validation'
        self._beta_full_validation = 'beta_full_validation'
        self._beta_s = 'beta_s'
        self._beta_full = 'beta_full'
        self._validation = '_validation'
        self._method='SLSQP'
        self._output = {}
        self._total_num_factors = np.array(self._num_factor_sectors).sum() + self._num_factor_region
        self._outputBeta = np.zeros((self._boundary[-1],self._total_num_factors))
        self._gamma = 'gamma'

    def getSectorNames(self, SID):
        return self._beta_s + str(SID)

    def getRegionName(self):
        return self._beta_region

    def getInitData(self):
        return self._initdata

    def getNumFactorRegion(self):
        return self._nuni_factor_region

    def getNumFactorSectors(self):
        return self._num_factor_sectors

    def getNumFactorRegion(self):
        return self._num_factor_region

    def loadInitData(self):
        xls = pd.ExcelFile(self._initdatafile)
        self._initdata[self._beta_region] = xls.parse(self._beta_region)
        for i in np.arange(len(self._num_factor_sectors)):
            self._initdata[self._beta_s + str(i)] = xls.parse(self._beta_s + str(i))

    def generateInitData(self,outputfile):
        #initdata = {}
        num_sectors = len(self._num_factor_sectors)
        num_factors = self._num_factor_region + np.array(self._num_factor_sectors).sum()
        num_rv = self._boundary[-1]
        initBeta_all = (np.random.rand(num_rv,self._num_factor_region)-0.5) * 2
        initBeta_sectors = [(np.random.rand(self._boundary[i+1]-self._boundary[i], self._num_factor_sectors[i]) - 0.5)*2  for i in np.arange(num_sectors)]
        if len(self._num_factor_sectors)>0:
            top = np.zeros((self._boundary[0], self._num_factor_sectors[0]))
            bottom = np.zeros((self._boundary[-1]-self._boundary[1], self._num_factor_sectors[0]))
            temp = np.bmat([[top], [initBeta_sectors[0]], [bottom]])
            initBeta_all = np.bmat([initBeta_all, temp])
            for i in np.arange(1, num_sectors):
                top = np.zeros((self._boundary[i], self._num_factor_sectors[i]))
                bottom = np.zeros((self._boundary[-1]-self._boundary[1+i], self._num_factor_sectors[i]))
                temp = np.bmat([[top], [initBeta_sectors[i]], [bottom]])
                initBeta_all = np.bmat([initBeta_all, temp])
        norm = np.sqrt( np.diag(initBeta_all.dot(initBeta_all.T)))
        norm = norm.reshape(num_rv,1)
        initBeta_all = initBeta_all/norm
        initBeta_all = np.array( initBeta_all) * np.sqrt(np.random.rand(num_rv,1))
        #outputfile = 'C:\\rk\\SFM\\input\\output_N50_r2_s23222_initdata.xisx'
        writer = pd.ExcelWriter(outputfile)
        self._initdata[self._beta_region] = pd.DataFrame(data = initBeta_all[:,0:self._num_factor_region])
        self._initdata[self._beta_region].to_excel(writer, sheet_name=self._beta_region)
        for i in np.arange(len(self._num_factor_sectors)):
            self._initdata[self._beta_s + str(i)] = pd.DataFrame(data = initBeta_all[self._boundary[i]:self._boundary[i+1], \
                np.int(np.array(self._num_factor_sectors[0:i]).sum()) \
                + self._num_factor_region : np.int( np.array(self._num_factor_sectors[0: i+1]).sum()) + self._num_factor_region])
            self._initdata[self._beta_s + str(i)].to_excel(writer, sheet_name =self._beta_s + str(i))
        writer.save()
        #return initdata

    def _XR_der_enhanced(self, B, C):
        n,m = C[self._shape]
        W = C[self._weight]
        U = C[self._selection]
        C_h = np.multiply(C[self._corr],W)
        B = B.reshape(n,m)
        B_h = np.multiply(B,U)
        temp = np.multiply(B_h.dot(B_h.T),W)
        temp = temp.dot(B_h)
        t1 = np.multiply(U, temp)
        t2 = np.multiply(C_h.dot(B_h),U)
        temp = np.multiply(np.diag(np.diag(W)),np.diag(np.diag(B_h.dot(B_h.T))))
        t3 = temp.dot(B_h)
        t4 = np.diag(np.diag(W)).dot(B_h)
        x = 4*(t1-t2-t3 +t4)
        return np.squeeze(np.array(x.reshape(1,n*m)))

    def _targetCorr_enhanced(self,B,C):
        n,m = C[self._shape]
        W = C[self._weight]
        U = C[self._selection]
        C_h = np.multiply(C[self._corr],W)
        B = B.reshape(n,m)
        B_h = np.multiply(B,U)
        x = np.ones((n,n)) - np.identity(n)
        W_h = np.multiply(x,W)
        temp = np.multiply(C_h-B_h.dot(B_h.T), C_h-B_h.dot(B_h.T))
        temp = np.multiply(temp, W_h)
        result = temp.sum()
        return result

    def _XR_der(self,B, C):
       n,m = C[self._shape]
       C = C[self._corr]
       B = B.reshape(n,m)
       x = 4*(B.dot(B.T.dot(B)) - np.diag(np.diag(B.dot(B.T))).dot(B) + B- C.dot(B))
       return np.squeeze(np.array(x.reshape(1,n*m)))

    def _targetCorr(self,B,C):
       n,m = C[self._shape]
       C = C[self._corr]
       B = B.reshape(n,m)
       xx = B.dot(B.T) - C + np.identity(n) - np.diag(np.diag(B.dot(B.T)))
       result = np.float(xx.dot(xx.T).trace())
       return result

    def calibration(self):
        num_sectors = len(self._num_factor_sectors)
        num_rv = self._boundary[-1]
        U = np.ones((num_rv, self._num_factor_region))
        W = np.ones(self._corrMatrix.shape)
        for i in np.arange(num_sectors):
            W[self._boundary[i]:self._boundary[i+1], self._boundary[i]:self._boundary[i+1]] = 0

        total_num_factors = np.array(self._num_factor_sectors).sum() + self._num_factor_region
        c0 = {self._shape: [num_rv,self._num_factor_region], self._corr:self._corrMatrix, self._weight:W, self._selection: U}
        beta_region = self._initdata[self._beta_region]
        m, n = beta_region.shape
        l = np.array(self._num_factor_sectors).sum()
        beta_region = beta_region.as_matrix()
        x0 = beta_region.reshape(num_rv*self._num_factor_region)
        bonds = tuple( (-1,1) for i in np.arange(num_rv*self._num_factor_region))
        funcs = [lambda x, i=i: self._upperbound_constrain-np.multiply(x[i*self._num_factor_region:((i+1)*self._num_factor_region)],\
                                                                       x[i*self._num_factor_region:((i+1)*self._num_factor_region)]).sum() for i in np.arange(num_rv)]
        connew = [{'type':'ineq', 'fun':funcs[i]} for i in np.arange(num_rv)]
        connew = (connew)
        res = minimize(self._targetCorr_enhanced, x0, method=self._method, jac=self._XR_der_enhanced, \
                       constraints= connew, bounds=bonds, options={'disp': True}, args = c0)
        beta = res.x.reshape(num_rv,self._num_factor_region)
        corr_region_part = beta.dot(beta.T)
        U, S = svd(corr_region_part)[0:2]
        beta = U[:,0:self._num_factor_region].dot(np.diag(S[0:self._num_factor_region]**0.5))
        outputBeta = np.zeros((num_rv,total_num_factors))
        outputBeta[:,0:self._num_factor_region] = beta


        self._output[self._beta_region] = pd.DataFrame(data = beta[:,0:self._num_factor_region], index = self._initdata[self._beta_region].index)
        self._output[self._beta_region_validation] = pd.DataFrame(data = beta[:,0:self._num_factor_region].dot(beta[:,0:self._num_factor_region].T)
                                                     - self._corrMatrix, index=self._initdata[self._beta_region].index, columns= self._initdata[self._beta_region].index)
        for i in np.arange(num_sectors):
            num_rv_s = self._boundary[i+1]-self._boundary[i]
            regioneffect = beta[self._boundary[i]:self._boundary[i+1], 0:self._num_factor_region].dot(beta[self._boundary[i]:self._boundary[i+1],0:self._num_factor_region].T)
            C_S = self._corrMatrix[self._boundary[i]:self._boundary[i+1], self._boundary[i]:self._boundary[i+1]] \
                  - beta[self._boundary[i]:self._boundary[i+1],0:self._num_factor_region].dot(beta[self._boundary[i]:self._boundary[i+1], 0:self._num_factor_region].T)
            C_S = C_S - np.diag(np.diag(C_S)) + np.identity(num_rv_s)
            c0_s = { self._shape:[num_rv_s, self._num_factor_sectors[i]], self._corr: C_S }
            beta_sector = self._initdata[self._beta_s + str(i)]
            beta_sector = beta_sector.as_matrix()
            x0_s = beta_sector.reshape(num_rv_s*self._num_factor_sectors[i])
            bounds = tuple( (-1, 1) for j in np.arange(self._num_factor_sectors[i]*(self._boundary[i+1]-self._boundary[i])) )
            funcs_sector = [lambda x, j=j: self._upperbound_constrain -regioneffect[j,j] \
                                           - np.multiply(x[(j*self._num_factor_sectors[i]):((j+1)*self._num_factor_sectors[i])],\
                                                         x[(j*self._num_factor_sectors[i]):((j+1)*self._num_factor_sectors[i])]).sum() for j in np.arange(num_rv_s)]
            connew_sector = [{'type': 'ineq', 'fun': funcs_sector[j]} for j in np.arange(num_rv_s)]
            connew_sector = (connew_sector)
            res_s =   minimize(self._targetCorr, x0_s, method=self._method, jac=self._XR_der, constraints= connew_sector, bounds=bounds, options={'disp': True}, args=c0_s)
            beta_s = res_s.x.reshape(num_rv_s, self._num_factor_sectors[i])
            U_S, S_S = svd(beta_s.dot(beta_s.T))[0:2]
            beta_s = U_S[:,0:self._num_factor_sectors[i]].dot(np.diag(S_S[0:self._num_factor_sectors[i]]**0.5))

            outputBeta[self._boundary[i]:self._boundary[i+1],\
                (self._num_factor_region+np.int(np.array(self._num_factor_sectors[0:i]).sum())) \
                :(self._num_factor_region+np.int(np.array(self._num_factor_sectors[0:i+1]).sum()))] = beta_s
            self._output[self._beta_s + str(i)] = pd.DataFrame(data=beta_s, index= self._initdata[self._beta_s+str(i)].index)
            self._output[self._beta_s + str(i) + self._validation] = pd.DataFrame(data=beta_s.dot(beta_s.T) - C_S , \
                index=self._initdata[self._beta_s + str(i)].index, columns= self._initdata[self._beta_s + str(i)].index)
            self._output[self._beta_s + str(i) + self._validation] = self._output[self._beta_s + str(i) + self._validation] \
                - np.diag(np.diag(self._output[self._beta_s + str(i) + self._validation]))
        self._output[self._beta_full] = pd.DataFrame(data = outputBeta,index= self._initdata[self._beta_region].index)
        self._output[self._beta_full_validation] = pd.DataFrame(data=outputBeta.dot(outputBeta.T) - self._corrMatrix, \
                            index= self._initdata[self._beta_region].index, columns=self._initdata[self._beta_region].index)
        self._output[self._beta_full_validation] = self._output[self._beta_full_validation] - np.diag(np.diag(self._output[self._beta_full_validation]))
        self._output[self._gamma] = pd.DataFrame(data = (1.0-np.array(np.diag( outputBeta.dot(outputBeta.T))))** 0.5)


    # factor model is Y = Beta * F + diag(gamma) * S
    # input: Y_samples_g x NI: Y samples for N-dim data, N= boundary[-1]. sample size =T
    # output: [ouputl, output2] outputl: T x M for factor scope, output2: T x 1 for idiosyncratic score
    def getFactorScoreByWLS(self, Y_samples): #weighted least square
        gamma = self._getGamma()
        if Y_samples.shape[0] != self._boundary[-1]:
            print('sample dim is %d, the number of variable is %d. they should be equal', (Y_samples.shape[0], self._boundary[-1]) )
            return np.nan
        beta = self._getBeta(self._beta_full)
        X = (beta.T.dot(np.diag(gamma**(-2.0)))).dot(beta) #M x M
        U = (beta.T.dot(np.diag(gamma**(-2.0)))).dot(Y_samples.T)
        Z = (X.I.dot(U)).T
        S = np.diag(gamma**(-1.0)).dot(Y_samples.T - beta.dot(Z))
        return [Z.T, S.T]

    def getGamma(self):
        return self._output[self._gamma]

    def getGlobalFactorLoading(self):
        return self._output[self._beta_region]

    def getAllFactorLoading(self):
        return self._output[self._beta_full]

    def getSectorFactorLoading(self, SID):
        return self._output[self._beta_s + str(SID)]

    def getAllValidation(self):
        return self._output[self._beta_full_validation]

    def getSectorValidation(self, SID):
        return self._output[self._beta_s + str(SID) + self._validation]

    def saveCalibrationOutput(self, outputfile):
        writer = pd.ExcelWriter(outputfile)
        self._output[self._beta_full].to_excel(writer, sheet_name=self._beta_full)
        self._output[self._beta_region].to_excel(writer, sheet_name=self._beta_region)
        self._output[self._beta_full_validation].to_excel(writer, sheet_name=self._beta_full_validation)
        self._output[self._gamma].to_excel(writer, sheet_name=self._gamma)
        for SID in np.arange(len(self._num_factor_sectors)):
            self._output[self._beta_s + str(SID)].to_excel(writer, sheet_name=self._beta_s + str(SID))
            self._output[self._beta_s + str(SID) + self._validation].to_excel(writer, sheet_name=self._beta_s + str(SID) + self._validation)
        writer.save()