import argparse
import logging
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import os
import sys
import pandas as pd
import numpy as np
import shutil
import math
from util.SampleConf import System as sample
#from util.ConfGenbyMM import ConfGen
from util.OptbySQM import System as sysopt
#from util.SPcalc import System as syssp
#from util.Align import Align as align
#from util.CalRMSD import RMSD
#from util.ConfRelaxbySQM import System as MDsample
from util.Cluster import cluster

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)


class binary_sample():
    def __init__(self, **args):
        try:
            self.db_name = args["input_sdf"]
        except Exception as e:
            self.db_name = None
        
        self.main_dir = os.getcwd()
        
        try:
            self.method = args["method"]
        except Exception as e:
            self.method = "SQM"

        try:
            self.energy_window = float(args["energy_window"])
        except Exception as e:
            self.energy_window = 20.0
        
        try:
            self.rmsd_cutoff = float(args["rmsd_cutoff"])
        except Exception as e:
            self.rmsd_cutoff = 2.0
         

    def sample_mm(self):
        ## sampling from vacuum

        if self.energy_window < 0:
            applied_MM_energy_window = 15
        else:
            applied_MM_energy_window = self.energy_window
        
        if self.rmsd_cutoff <= 0:
            applied_MM_rmsd_cutoff = 1.0
        else:
            applied_MM_rmsd_cutoff = self.rmsd_cutoff


        sample(input_sdf=self.db_name,
                type=1,
                save_frame=350,
                mm_energy_window=applied_MM_energy_window,
                mm_rmsd_cutoff=applied_MM_rmsd_cutoff).run()
        
        if not (os.path.isfile("SAVE.sdf") and os.path.getsize("SAVE.sdf")):
            logging.info("Failed at MM sampling, abort")
            return None
        
        
        reduced = cluster(input_sdf="SAVE.sdf",
                            rmsd_cutoff_cluster=self.rmsd_cutoff,
                            do_align=True,
                            only_reduce_duplicate=True).run()
        
        ## perform opt at h2o level
        opted_hoh = sysopt(input_rdmol_obj=reduced, 
                            rmsd_cutoff=self.rmsd_cutoff,
                            # save_n=self.sp_max_n,
                            HA_constrain=False,
                            if_write_sdf=False).run()
        
        sorted_opted_hoh = sorted(opted_hoh, key=lambda x: float(x.GetProp("Energy_xtb")))
        if self.energy_window < 0:
            saved_opt_hoh = sorted_opted_hoh
        else:
            saved_opt_hoh = [cc for cc in sorted_opted_hoh \
                            if (float(cc.GetProp("Energy_xtb")) - float(sorted_opted_hoh[0].GetProp("Energy_xtb")))* 627.51 <= self.energy_window]
        
        opted_chcl3 = sysopt(input_rdmol_obj=reduced, 
                            rmsd_cutoff=self.rmsd_cutoff,
                            # save_n=self.sp_max_n,
                            HA_constrain=False,
                            solvation="chcl3",
                            if_write_sdf=False).run()
        
        sorted_opted_chcl3 = sorted(opted_chcl3, key=lambda x: float(x.GetProp("Energy_xtb")))

        if self.energy_window < 0:
            saved_opted_chcl3 = sorted_opted_chcl3
        else:
            saved_opted_chcl3 = [cc for cc in sorted_opted_chcl3 \
                         if (float(cc.GetProp("Energy_xtb")) - float(sorted_opted_chcl3[0].GetProp("Energy_xtb")))* 627.51 <= self.energy_window]

        os.system("rm -f SAVE.sdf")

        return saved_opt_hoh,saved_opted_chcl3

    def sample_sqm(self):
        ## sample in hoh solvation and perform opt
        sample(input_sdf=self.db_name,
                type=2,
                run_temperature=350,
                save_frame=350).run()
        
        if not (os.path.isfile("SAVE.sdf") and os.path.getsize("SAVE.sdf")):
            logging.info("Failed at SQM_hoh sampling, abort")
            return None
        
        reduced = cluster(input_sdf="SAVE.sdf",
                            rmsd_cutoff_cluster=self.rmsd_cutoff,
                            do_align=True,
                            only_reduce_duplicate=True).run()
            
        ## perform opt at h2o level
        opted_hoh = sysopt(input_rdmol_obj=reduced, 
                       rmsd_cutoff=self.rmsd_cutoff,
                      # save_n=self.sp_max_n,
                       HA_constrain=True,
                       if_write_sdf=False).run()
        
        sorted_opted_hoh = sorted(opted_hoh, key=lambda x: float(x.GetProp("Energy_xtb")))

        if self.energy_window < 0:
            saved_opt_hoh = sorted_opted_hoh
        else:
            saved_opt_hoh = [cc for cc in sorted_opted_hoh \
                         if (float(cc.GetProp("Energy_xtb")) - float(sorted_opted_hoh[0].GetProp("Energy_xtb")))* 627.51 <= self.energy_window]
        
        os.system("rm -f SAVE.sdf")

        ## sample in hoh solvation and perform opt
        sample(input_sdf=self.db_name,
                type=2,
                run_temperature=350,
                save_frame=350,
                solvation="chcl3").run()
                
        
        if not (os.path.isfile("SAVE.sdf") and os.path.getsize("SAVE.sdf")):
            logging.info("Failed at SQM_chcl3 sampling, abort")
            return None
        
        reduced_chcl3 = cluster(input_sdf="SAVE.sdf",
                            rmsd_cutoff_cluster=self.rmsd_cutoff,
                            do_align=True,
                            only_reduce_duplicate=True).run()
        
        opted_chcl3 = sysopt(input_rdmol_obj=reduced_chcl3, 
                            rmsd_cutoff=self.rmsd_cutoff,
                            # save_n=self.sp_max_n,
                            HA_constrain=True,
                            solvation="chcl3",
                            if_write_sdf=False).run()
        
        sorted_opted_chcl3 = sorted(opted_chcl3, key=lambda x: float(x.GetProp("Energy_xtb")))

        if self.energy_window < 0:
            saved_opted_chcl3 = sorted_opted_chcl3
        else:
            saved_opted_chcl3 = [cc for cc in sorted_opted_chcl3 \
                         if (float(cc.GetProp("Energy_xtb")) - float(sorted_opted_chcl3[0].GetProp("Energy_xtb")))* 627.51 <= self.energy_window]

        os.system("rm -f SAVE.sdf")
        
        return saved_opt_hoh, saved_opted_chcl3
    
    def sample(self):
        _dic = {"MM": self.sample_mm,
                "SQM": self.sample_sqm}
        
        try:
            _dic[self.method]
        except Exception as e:
            logging.info("Wrong method type, choose from ['SQM', 'MM'] and run again, abort")
            return 
        
        if not self.db_name:
            logging.info("No input, nothing to do, abort")
            return 
        
        prefix = ".".join(self.db_name.split(".")[:-1])

        work_dir = os.path.join(self.main_dir, f"{prefix}_{self.method}")

        if not os.path.exists(work_dir):
            os.mkdir(work_dir)
        
        os.chdir(work_dir)
        target = os.path.join(self.main_dir, self.db_name)
        os.system(f"mv {target} {work_dir}")

        saved_opt_hoh, saved_opted_chcl3 = _dic[self.method]()

        return saved_opt_hoh, saved_opted_chcl3

def lbg_api(input_sdf, method, rmsd_cutoff, energy_window):
    
    main_dir = os.getcwd()

    try:
        mol_db = [mm for mm in Chem.SDMolSupplier(input_sdf, removeHs=False) if mm]
    except Exception as e:
        logging.info("Wrong input sdf file, abort")
        return 
    
    if not mol_db:
        logging.info("Wrong input sdf file, abort")
        return
    
    name_tag = [mm.GetProp("_Name") for idx, mm in enumerate(mol_db)]

    if len(set(name_tag)) != len(mol_db):
        assign_name = [".".join(input_sdf.split(".")[:-1])+"_"+str(i) for i in range(len(mol_db))]
    else:
        assign_name = name_tag
    
    #_dic = {}
    
    for idx, mm in enumerate(mol_db):
        logging.info(f"Working with {assign_name[idx]}...")

        cc = Chem.SDWriter(f"{assign_name[idx]}.sdf")
        cc.write(mm)
        cc.close()

        sample_hoh, sample_chcl3 = binary_sample(input_sdf=f"{assign_name[idx]}.sdf", 
                                                    method=method,
                                                    rmsd_cutoff=rmsd_cutoff,
                                                    energy_window=energy_window).sample()

        cc_hoh = Chem.SDWriter("sampled_hoh.sdf")
        for each in sample_hoh:
            cc_hoh.write(each)
        cc_hoh.close()

        cc_chcl3 = Chem.SDWriter("sampled_chcl3.sdf")
        for every in sample_chcl3:
            cc_chcl3.write(every)
        cc_chcl3.close()

        os.chdir(main_dir)

        #with open("sampled_hoh.sdf", "r+") as hoh:
        #    content_hoh = [cc for cc in hoh.readlines()]
        
        #with open("sampled_chcl3.sdf", "r+") as chcl3:
        #    content_chcl3 = [dd for dd in chcl3.readlines()]
        
        #_dic.setdefault(f"{assign_name[idx]}_{method}", [content_hoh, content_chcl3])

        #logging.info(f"Sampling done for {assign_name[idx]}")
    
    return 


def app_api(input_sdf, method, rmsd_cutoff, energy_window):
    
    main_dir = os.getcwd()

    try:
        mol_db = [mm for mm in Chem.SDMolSupplier(input_sdf, removeHs=False) if mm]
    except Exception as e:
        logging.info("Wrong input sdf file, abort")
        return 
    
    if not mol_db:
        logging.info("Wrong input sdf file, abort")
        return
    
    name_tag = [mm.GetProp("_Name") for idx, mm in enumerate(mol_db)]

    if len(set(name_tag)) != len(mol_db):
        assign_name = [".".join(input_sdf.split(".")[:-1])+"_"+str(i) for i in range(len(mol_db))]
    else:
        assign_name = name_tag
    
    _dic = {}
    
    for idx, mm in enumerate(mol_db):
        logging.info(f"Working with {assign_name[idx]}...")
        cc = Chem.SDWriter(f"{assign_name[idx]}.sdf")
        cc.write(mm)
        cc.close()

        sample_hoh, sample_chcl3 = binary_sample(input_sdf=f"{assign_name[idx]}.sdf", 
                                                    method=method,
                                                    rmsd_cutoff=rmsd_cutoff,
                                                    energy_window=energy_window).sample()

        cc_hoh = Chem.SDWriter("sampled_hoh.sdf")
        for each in sample_hoh:
            cc_hoh.write(each)
        cc_hoh.close()

        cc_chcl3 = Chem.SDWriter("sampled_chcl3.sdf")
        for every in sample_chcl3:
            cc_chcl3.write(every)
        cc_chcl3.close()

        with open("sampled_hoh.sdf", "r+") as hoh:
            content_hoh = [cc for cc in hoh.readlines()]
        
        with open("sampled_chcl3.sdf", "r+") as chcl3:
            content_chcl3 = [dd for dd in chcl3.readlines()]
        
        _dic.setdefault(f"{assign_name[idx]}_{method}", [content_hoh, content_chcl3])

        logging.info(f"Sampling done for {assign_name[idx]}")
    
    os.chdir(main_dir)
    
    return _dic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tailored stable conformer generation workflow for MacroMolecule, \
                                    $WORKDIR has same name with input sdf, \
                                    final saving include "sampled_hoh.sdf" and "sampled_chcl3.sdf" \
                                    stable conformation ensemble in water and membrane')
    parser.add_argument('--input_sdf', type=str, required=True, 
                        help='input sdf file, docking pose(s) for single mol')
    parser.add_argument('--method', type=str, default="SQM",
                        help='sampling method, choose from ["SQM", "MM"], default SQM')
    parser.add_argument('--rmsd_cutoff', type=float, default=2.0,
                        help='rmsd cutoff to reduce redundancy, default 2.0')
    parser.add_argument('--energy_window', type=float, default=20.0,
                        help='energy window applied, default 20.0')
    
    args = parser.parse_args()

    lbg_api(input_sdf=args.input_sdf, 
                  method=args.method,
                  rmsd_cutoff=args.rmsd_cutoff,
                  energy_window=args.energy_window)