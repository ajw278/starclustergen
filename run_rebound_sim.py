import rebound
import csv
import ctypes
import numpy as np
import copy
import sys
import os

class enctrack():
        def __init__(self, **kwargs):
            self.closestars = {}
            

class setupSimulation():
    def __init__(self, r, v, m, units=('Myr', 'pc', 'Msun'), fname='archive', dtsave=1e-4):
        self.r = r
        self.v = v
        self.m = m
        self.units = units
        
        self.csvfile = "test.csv"
        if not os.path.isfile(fname+'.bin'):
            with open(self.csvfile, "w") as csvfile: 
                writer = csv.writer(csvfile)

                # write down columns_name
                writer.writerow(["n_encounter","time(Myr)","i1","i2","eccentricity","distance_closest(pc)","sim.d(pc)","m1(m_sun)","m2(m_sun)"])


            self.n_encounter = 0
            self.sim = rebound.Simulation()
            self.sim.units = self.units
            self.sim.integrator = "ias15" # IAS15 is the default integrator
            self.sim.collision = "direct"

            def encounter_record(a, b):
                return self.encounter_record(self, a, b)
            self.sim.collision_resolve = encounter_record

            self.N = self.r.shape[1]
            for i in range(self.N):
                self.sim.add(m=m[i], x=r[0,i], y=r[1,i], z=r[2,i], vx=v[0,i], vy=v[1,i], vz=v[2,i], r = 0.01)#r_lim_pc.value)
                self.sim.particles[i].ap = ctypes.c_void_p(1)
            self.sim.move_to_com()
            self.sim.save_to_file(fname+".bin", interval=dtsave)
        else:
            self.sim = rebound.Simulation("archive.bin", snapshot=-1)
            self.sim.save_to_file(fname+".bin", interval=dtsave)
        
        return None
    
    def integrate(self, time):
        self.sim.integrate(time)
    
    #Calculate eccentricity and closest approach of encounter
    def calc_e_rp(self, dr, dv, m1, m2, hh):

        mu = self.sim.G * (m1 + m2)
        a = 1 / (2 / dr - dv * dv / mu)
        v_infinity = np.sqrt(abs( -mu / a))
        b = hh / v_infinity

        if a > 0:
            ee = np.sqrt(1 - b**2 / a**2)
        else:
            ee = np.sqrt(1 + b**2 / a**2)

        rr_p = -a * (ee - 1)

        return ee, rr_p
    
    #Calculate specific angular momentum
    def calc_h(self, dx, dy, dz, dvx, dvy, dvz):

        dr = np.array([dx, dy, dz])
        dv = np.array([dvx, dvy, dvz])

        hh = np.cross(dr, dv)

        return np.linalg.norm(hh) 

    #Take the dot product of the position and velocity vectors
    def calc_r_v(self, dx, dy, dz, dvx, dvy, dvz):

        dr = np.array([dx, dy, dz])
        dv = np.array([dvx, dvy, dvz])
        dr_dv = np.dot(dr, dv) #dr @ dv
    #     print(np.dot(dr, dv), dr @ dv, dr, dv)

        if dr_dv > 0: #away
            return 0
        else:         #towards
            return 1


    @staticmethod
    def encounter_record(self, sim_pointer, collision):

        sim = sim_pointer.contents           # get simulation object from pointer

        i1 = sim.particles[collision.p1].index
        i2 = sim.particles[collision.p2].index
        m1 = sim.particles[collision.p1].m
        m2 = sim.particles[collision.p2].m

        # Calculates the coponentwise difference between particles 
        dp = sim.particles[i1] - sim.particles[i2] 

        #The distance between two particles in 3D space
        delta_r = sim.particles[collision.p1] ** sim.particles[collision.p2] 
        #np.sqrt(dp.x*dp.x+dp.y*dp.y+dp.z*dp.z)

        N = len(sim.particles)
        #check whether they are the closest pair of stars
        kk1 = N + 1

        dd = self.calc_r_v(dp.x, dp.y, dp.z, dp.vx, dp.vy, dp.vz) # 0(away) or 1(towards)

        for k1 in range(N):
                delta_r1 = sim.particles[int(k1)] ** sim.particles[i1]
                if delta_r1 < delta_r and int(k1)!= i1:
                    kk1 = k1
        
        if kk1 == N+1:

            #Is this a change of direction:
            #If ap == 1 in the previous timestep then the two stars were moving towards each other
            #If dd == 0 in this time-step, then they are now moving away from each other
            ca = (sim.particles[collision.p1].ap == 1)&(dd == 0)

            sim.particles[collision.p1].ap = copy.copy(dd) 

            #Store close encounter if this is the closest time-step
            if ca:
                delta_v = np.sqrt(dp.vx * dp.vx + dp.vy * dp.vy + dp.vz * dp.vz)
                self.n_encounter += 1
                h = self.calc_h(dp.x, dp.y, dp.z, dp.vx, dp.vy, dp.vz)
                e, r_p = self.calc_e_rp(delta_r, delta_v, m1, m2, h)

                with open(self.csvfile,'a+') as f: # 'a' indicates do not overwritting the previous content 
                    csv_write = csv.writer(f)
                    data_row = [self.n_encounter, sim.t, i1, i2, e, r_p, delta_r, m1, m2]
                    # data_row = [n_encounter, sim.t, i1, i2, e, e_p1, e_p2, r_p, delta_r, m1, m2]
                    csv_write.writerow(data_row)


        return 0