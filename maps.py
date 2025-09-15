import numpy as np

class Maps:
    
    def _init__(self, Data=None):
        
        if Data is not None:
            self.Data = Data
            
    def Half_Amp_Latency(self, Trace):
        '''
        Calculate the Half Amplitude Latency of a given Trace (1-D). The final results are the moments after the start point of the time window.
        '''
        
        max_amplitude = np.max(Trace)
        max_index = np.argmax(Trace)
        value = 0
        
        for k in range(0, max_index):
            
            if 0==max_index:
                value = 0
                break
            elif Trace[k] == max_amplitude/2:
                value = k*0.5
                break
            elif Trace[k] > max_amplitude/2 and k==0:
                value = (max_amplitude/2)/Trace[k] * 0.5
                break
            elif Trace[k] > max_amplitude/2 and k!=0:
                value = (k-1)*0.5 + (max_amplitude/2 - Trace[k-1])/(Trace[k] - Trace[k-1])*0.5
                break
            elif k==max_index-2 and Trace[k] < max_amplitude/2:
                value = k*0.5 + (max_amplitude/2 - Trace[k])/(max_amplitude - Trace[k])*0.5
                break
            
        return value
