class Output:
    def __init__(self,
        nout = 100,             # Time step to save results
        filepath = 'test',      # filepath to save results                    
        T_stat = True,          # Generate time histories at specific points  
        L_save = True,          # Save results
        nsensors = 0,
    ):
        self.nout = nout
        self.filepath = filepath
        self.T_stat = T_stat
        self.L_save = L_save
        self.nsensors = nsensors
        self.sensor_locations = np.zeros((2, nsensors))
        # self.sensor_locations = np.zeros((ntimesteps, nsensors))

    def default_sensor_locations(nsensors):
        return None

    def store(self,
        itime,
        prognostic_variables,
    ):
        if self.L_save and itime % self.nout == 0:

            u = prognostic_variables.u
            v = prognostic_variables.v

            # np.save(self.filepath+"/u"+str(itime), arr=u.cpu().detach().numpy())
            # np.save(self.filepath+"/v"+str(itime), arr=v.cpu().detach().numpy()) 
            
            # if self.T_stat == True:
            #     for k in range(self.nsensors):
            #         self.sensor_locations[k, itime-1] = u[0,0,p_y[k],p_x[k]] 