import numpy as np

class Tools:
    
    def __init__(self, Data=None):
        '''
        Load the Data that are in a shape of Trials * height * width * points.
        '''
        if Data is None:
            pass
        else:
            self.Data = Data
        
    def Polynomial(self, startPt=None, numPt=None, Data=None):
        '''
        Implement 3-degrees polynomial regression to the Original Data. The skip window can be set by adjust numPt and startPt.
        '''
        
        if Data is None:
            Data = self.Data
        
        index = np.linspace(0, Data.shape[3] * 0.5, Data.shape[3])
        
        if startPt is not None:
            if numPt is not None:
                index_d = np.delete(index, np.arange(startPt, startPt+numPt, step=1), axis=0)
                index_skip = True
        else:
            index_d = index
            index_skip = False
          
        for i in range(Data.shape[0]):
            for j in range(Data.shape[1]):
                for k in range(Data.shape[2]):
                    
                    if index_skip:
                        array = np.delete(np.copy(Data[i][j][k]), np.arange(startPt, startPt+numPt, step=1), axis=0)
                    else:
                        array = np.copy(Data[i][j][k])
                    
                    coeffs = np.polyfit(index_d, array, 3)
                    poly = np.poly1d(coeffs)

                    Data_fit = poly(index)
                    Data[i][j][k] = Data[i][j][k] - Data_fit
                    
        return Data
    
    def T_filter(self, Data=None):
        '''
        Apply bionomial8 Temporal Filter to the Target Data.
        '''
        
        if Data is None:
            Data = self.Data
        
        for i in range(Data.shape[0]):
                    
            input = np.copy(Data[i])
        
            num = Data.shape[3]
            output = np.zeros((Data.shape[1], Data.shape[2], Data.shape[3]))

            output[:, :, 0] = (input[:, :, 4] + 8 * input[:, :, 3] + 28 * input[:, :, 2] + 56 * input[:, :, 1]) / 93
            output[:, :, 1] = (input[:, :, 5] + 8 * input[:, :, 4] + 28 * input[:, :, 3] + 56 * input[:, :, 2] 
                               + 70 * input[:, :, 1]) / 163
            output[:, :, 2] = (input[:, :, 6] + 8 * input[:, :, 5] + 28 * input[:, :, 4] 
                               + 56 * (input[:, :, 1] + input[:, :, 3]) + 70 * input[:, :, 2]) / 219
            output[:, :, 3] = (input[:, :, 7] + 8 * input[:, :, 6] + 28 * (input[:, :, 1] + input[:, :, 5]) 
                               + 56 * (input[:, :, 2] + input[:, :, 4]) + 70 * input[:, :, 3]) / 247

            end = num-4	
            for s in range(4, end):
                output[:, :, s] = (input[:, :, s-4]+input[:, :, s+4] + 8*(input[:, :, s-3]+input[:, :, s+3]) 
                                    + 28*(input[:, :, s-2]+input[:, :, s+2])+56*(input[:, :, s-1]+input[:, :, s+1])+70*input[:, :, s])/256

            output[:, :, num-4] = (input[:, :, num-8] + 8 * (input[:, :, num-7] + input[:, :, num-1]) 
                                    + 28 * (input[:, :, num-6] + input[:, :, num-2]) + 56 * (input[:, :, num-5] + input[:, :, num-3]) 
                                    + 70 * input[:, :, num-4]) / 255
            output[:, :, num-3] = (input[:, :, num-7] + 8 * input[:, :, num-6] + 28 * (input[:, :, num-5] + input[:, :, num-1]) 
                                    + 56 * (input[:, :, num-4] + input[:, :, num-2]) + 70 * input[:, :, num-3]) / 247
            output[:, :, num-2] = (input[:, :, num-6] + 8 * input[:, :, num-5] + 28 * input[:, :, num-4] 
                                    + 56 * (input[:, :, num-3] + input[:, :, num-1]) + 70 * input[:, :, num-2]) / 219
            output[:, :, num-1] = (input[:, :, num-5] + 8 * input[:, :, num-4] + 28 * input[:, :, num-3] 
                                    + 56 * input[:, :, num-2] + 70 * input[:, :, num-1]) / 163
            
            '''
            output[0] = (output[6]+output[10])/2
            output[1] = (output[7] + output[9]) / 2
            output[2] = (output[8] + output[10]) / 2
            output[3] = (output[9] + output[6]) / 2
            output[4] = (output[10] + output[7]) / 2
            output[5] = (output[11] + output[13]) / 2
            '''
            
            Data[i] = output
                
        return Data
    
    def S_filter(self, sigma, Data=None):
        '''
        Apply Gaussian Spatial Filter to the Target Data. Sigma here means the spatial constant which is used to calculate the ceterWeight.
        '''
        
        if Data is None:
            Data = self.Data
        
        Trials, height, width, points = Data.shape    
        proData = np.zeros((Trials, height*width, points))
        for i in range(Data.shape[0]):
            proData[i] = Data[i].reshape(height*width, points)
            
        centerWeight = np.exp((0.5)/(sigma**2))
        
        num_diodes, numPts = proData.shape[1], proData.shape[2]
        output = np.zeros_like(proData)

        for i in range(Data.shape[0]):
            for center_diode in range(num_diodes):
                
                x = center_diode % width
                y = center_diode // width

                acc = np.zeros(numPts)
                weight_sum = 0.0

                acc += proData[i][center_diode] * centerWeight
                weight_sum += centerWeight

                for j in range(9):
                    if j == 4:
                        continue

                    xoffset = (j % 3) - 1
                    yoffset = (j // 3) - 1
                    nx, ny = x + xoffset, y + yoffset

                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue

                    neighbor = nx + ny * width
            

                    acc += proData[i][neighbor]
                    weight_sum += 1.0

                if weight_sum > 0:
                    output[i][center_diode] = acc / weight_sum
                else:
                    output[i][center_diode] = 0.0
        
        output = output.reshape(Data.shape[0], Data.shape[1], Data.shape[2], Data.shape[3])
        
        return output