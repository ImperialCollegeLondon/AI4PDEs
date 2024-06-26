class Simulation:
    def __init__(self,
        prognostic_variables,
        diagnostic_variables,
        model,
    ):
        self.prognostic_variables = prognostic_variables
        self.diagnostic_variables = diagnostic_variables
        self.model = model

    def run(self,
        ntimesteps = 100):
        pass
        # with torch.no_grad():
        # for itime in range(1,ntimesteps+1):

        # if itime % self.model.feedback.ncheck == 0:
        #     print('Time step:', itime, 'Pressure residual:',"{:.5f}".format(np.max(np.abs(w.cpu().detach().numpy()))))  
        # if np.max(np.abs(w.cpu().detach().numpy())) > 80000.0:
        #     print('Not converged !!!!!!')
        #     break
        # if L_save and itime % nout == 0:
        #     np.save(filepath+"/u"+str(itime), arr=values_u.cpu().detach().numpy())
        #     np.save(filepath+"/v"+str(itime), arr=values_v.cpu().detach().numpy()) 
        # if T_stat == True:
        #     for k in range(N_p):
        #         num_p[k,itime-1] = values_u[0,0,p_y[k],p_x[k]] 
        # end = time.time()
        # print('Elapsed time:', end - start)