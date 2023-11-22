import rebound
import ctypes


#Calculate eccentricity and closest approach of encounter
def calc_e_rp(dr, dv, m1, m2, hh):
    
    mu = sim.G * (m1 + m2)
    a = 1 / (2 / dr - dv * dv / mu)
    v_infinity = np.sqrt(abs( -mu / a))
    b = hh / v_infinity
    
    if a > 0:
        ee = np.sqrt(1 - b**2 / a**2)
    else:
        ee = np.sqrt(1 + b**2 / a**2)
        
    rr_p = -a * (ee - 1)
#     pp = np.sqrt(abs(4 * np.pi * a**3 / mu))
    
    return ee, rr_p

#Calculate specific angular momentum
def calc_h(dx, dy, dz, dvx, dvy, dvz):
    
    dr = np.array([dx, dy, dz])
    dv = np.array([dvx, dvy, dvz])
    
    hh = np.cross(dr, dv)
    
    return np.linalg.norm(hh) 

#Take the dot product of the position and velocity vectors
def calc_r_v(dx, dy, dz, dvx, dvy, dvz):
    
    dr = np.array([dx, dy, dz])
    dv = np.array([dvx, dvy, dvz])
    dr_dv = np.dot(dr, dv) #dr @ dv
#     print(np.dot(dr, dv), dr @ dv, dr, dv)
    
    if dr_dv > 0: #away
        return 0
    else:         #towards
        return 1
    

path_csv = "test.csv"
with open(path_csv, "w") as csvfile: 
    writer = csv.writer(csvfile)

    # write down columns_name
    writer.writerow(["n_encounter","time(Myr)","i1","i2","eccentricity","distance_closest(pc)","sim.d(pc)","m1(m_sun)","m2(m_sun)"])

    
n_encounter = 0
def encounter_record(sim_pointer, collision):
    
    global n_encounter
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

    #check whether they are the closest pair of stars
    kk1 = N + 1
    
    dd = calc_r_v(dp.x, dp.y, dp.z, dp.vx, dp.vy, dp.vz) # 0(away) or 1(towards)
    
    for k1 in range(N):
            delta_r1 = sim.particles[int(k1)] ** sim.particles[i1]
            if delta_r1 < delta_r and int(k1)!= i1:
                kk1 = k1
    # print(i1, i2, kk1, sim.particles[collision.p1].ap, dd)

        
    # p2 is the closest star to p1
    if kk1 == N+1:
    
        #Is this a change of direction:
        #If ap == 1 in the previous timestep then the two stars were moving towards each other
        #If dd == 0 in this time-step, then they are now moving away from each other
        ca = (sim.particles[collision.p1].ap == 1)&(dd == 0)
        
        sim.particles[collision.p1].ap = copy.copy(dd) 
        
        #Store close encounter if this is the closest time-step
        if ca:
            delta_v = np.sqrt(dp.vx * dp.vx + dp.vy * dp.vy + dp.vz * dp.vz)
            n_encounter += 1
            h = calc_h(dp.x, dp.y, dp.z, dp.vx, dp.vy, dp.vz)
            e, r_p = calc_e_rp(delta_r, delta_v, m1, m2, h)
            e_p1 = sim.particles[i1].calculate_orbit(primary = sim.particles[i2])
            e_p2 = sim.particles[i2].calculate_orbit(primary = sim.particles[i1])
            #print(n_encounter)

            with open(path_csv,'a+') as f: # 'a' indicates do not overwritting the previous content 
                csv_write = csv.writer(f)
                data_row = [n_encounter, sim.t, i1, i2, e, r_p, delta_r, m1, m2]
                # data_row = [n_encounter, sim.t, i1, i2, e, e_p1, e_p2, r_p, delta_r, m1, m2]
                csv_write.writerow(data_row)
            
        
#         sim.particles[collision.p1].ap = ctypes.c_void_p(dd)
#         sim.particles[collision.p2].ap = ctypes.c_void_p(dd)

        #Temporary to check if works:
        #sim.particles[collision.p1].ap = *{collision.p2: dd}

        #NOT CORRECT, BUT THE IDEA - INITIATE ap AS PYTHON DICT BEFORE RUNNING CODE
        #Is this change of direction:
#         if collision.p2 in sim.particles[collision.p1].ap:
#             ca = (sim.particles[collision.p2].ap==1)&(dd==0)


    return 0

class enctrack():
        def __init__(self, **kwargs):
            self.closestars = {}
            

def setupSimulation(xyz, vxyz, m):
    sim = rebound.Simulation()
    sim.units = ('Myr', 'pc', 'Msun')
    sim.integrator = "ias15" # IAS15 is the default integrator
    sim.collision = "direct"
    sim.collision_resolve = encounter_record
    for i in range(N):
        sim.add(m=m[i], x=xyz[i,0], y=xyz[i,1], z=xyz[i,2], vx=vxyz[i,0], vy=vxyz[i,1], vz=vxyz[i,2], r = 0.1)#r_lim_pc.value)
        sim.particles[i].ap = ctypes.c_void_p(1)
    sim.move_to_com()
    
    return sim