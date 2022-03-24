import os 
import sys 
import pandas as pd


sys.path.append('/mnt/mnemo5/tao/MLG/task_1/')
from utils.data_loader import load_train_genes_for_cell_line


data_small_folder="/mnt/mnemo5/tao/MLG/task_1/data_small"



def created_gct_file_train_plus_val() -> pd.DataFrame :
    
    """
    create gct files for cell line 1 and cell 2 (train and val data merged)
    
    :return: Dictionary contain gct file (pd.DataFrame format)
    
    """
    GCT_frame_dict={}
    for cell_line in [1, 2]:
        # notice  here train_genes is combinationo of train and val data in original dataset 
        merged_train_info=load_train_genes_for_cell_line(cell_line=cell_line)
        merged_train_list=merged_train_info.loc[:,["gene_name","chr","gene_start","gene_end","gex"]].values.tolist()
        
        GCT_list=list()
        for gene_name,cchr,gene_start,gene_end ,gex,in merged_train_list:
            GCT_list.append([gene_name,
                             " na |@"+cchr+":"+str(gene_name)+"-"+str(gene_end)+"|",
                             gex,])
        GCT_frame_dict[cell_line]=pd.DataFrame(GCT_list,columns=["name","Description","sample 1"])

    return(GCT_frame_dict)

if __name__ == '__main__':
    
    GCT_frame_dict=created_gct_file_train_plus_val()
    
    for key in GCT_frame_dict.keys():
        GCT_frame_dict[key].to_csv(os.path.join(data_small_folder,"X"+str(key)+"_trainAndVal.gct"),header=True,index=None,sep="\t")