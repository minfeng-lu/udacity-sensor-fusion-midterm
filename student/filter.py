# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.dim_state = params.dim_state
        self.dt = params.dt
        self.q = params.q

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        dim_state = self.dim_state
        dt = self.dt
        F = np.identity(dim_state).reshape(dim_state, dim_state)
        F[0, 3] = 1
        F[1, 4] = 1
        F[2, 5] = 1
        return np.matrix(F)
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        q = self.q
        dim_state = self.dim_state
        dt = self.dt
        q3t = ((dt**3)/3) * q 
        q2t = ((dt**2)/2) * q 
        qt = dt * q 
        Q = np.zeros((dim_state, dim_state))
        Q[0, 0] = q3t
        Q[1, 1] = q3t
        Q[2, 2] = q3t
        Q[0, 3] = q2t
        Q[1, 4] = q2t
        Q[2, 5] = q2t
        Q[3, 0] = q2t
        Q[4, 1] = q2t
        Q[5, 2] = q2t   
        Q[3, 3] = qt    
        Q[4, 4] = qt    
        Q[5, 5] = qt    
        return Q
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        F = self.F()
        x = F * track.x # state prediction
        P = F * track.P* F.transpose() + self.Q() # covariance prediction 
        track.set_x(x)
        track.set_P(P)       
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        H = meas.sensor.get_H(track.x) # measurement matrix
        gamma = self.gamma(track, meas) # residual
        P = track.P
        S = self.S(track, meas, H) # covariance of residual
        K = P * H.transpose() * np.linalg.inv(S) # Kalman gain
        x = track.x + K * gamma # state update
        I = np.identity(self.dim_state)
        P = (I - K * H) * P # covariance update
        track.set_x(x)
        track.set_P(P) 
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        gamma = meas.z - meas.sensor.get_hx(track.x)
        return gamma
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        S = H * track.P * H.transpose() + meas.R
        return S
        
        ############
        # END student code
        ############ 