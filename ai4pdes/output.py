class Output:
    def __init__(self,
        nout = 100,             # Time step to save results
        filepath = 'test',      # filepath to save results                    
        T_stat = True,          # Generate time histories at specific points  
        L_save = True,          # Save results
    ):
        self.nout = nout
        self.filepath = filepath
        self.T_stat = T_stat
        self.L_save = L_save