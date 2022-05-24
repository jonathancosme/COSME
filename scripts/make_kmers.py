from global_funcs import load_data_config
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import numpy as np

if __name__ == '__main__':
    import dask_cudf

    print("loading configs...")
    configs = load_data_config()
    
    print("setting variables...")
    input_col_name = configs['input_col_name']
    data_dir = configs['data_dir']
    k_mer = configs['k_mer']
    dask_dir = configs['dask_dir']
    possible_gene_values = configs['possible_gene_values']
    possible_gene_values = sorted(possible_gene_values)
    
    print("starting dask cluster...")
    cluster = LocalCUDACluster(local_directory=dask_dir)
    client = Client(cluster)
    print("finished starting dask cluster.")

    # max_k_mer = 12
    
    replace_gene_values = []
    for gene_val in possible_gene_values:
        replace_gene_values.append(gene_val + ' ')
        
    def add_whitespace(df):
        df[input_col_name] = df[input_col_name].str.replace(possible_gene_values, replace_gene_values, regex=False)
        return df
    
    def get_kmers(df):
        df['temp'] = df[input_col_name].copy()
        df['temp'] = ' ' 
        for i in np.arange(0, df[input_col_name].str.len().max() - k_mer):
            # print(i)
            temp_df = df[input_col_name].str[i: i+k_mer].fillna(' ')
            change_mask = temp_df.str.len() < k_mer
            temp_df[change_mask] = ' ' 
            df['temp'] = df['temp'] + ' ' + temp_df  
        df['temp'] = df['temp'].str.normalize_spaces()
        df[input_col_name] = df['temp']
        df = df.drop(columns=['temp'])
        return df
    
    def split_whitespace(df):
        df[input_col_name] = df[input_col_name].str.split()
        return df

    print(f"reading data from:\n{data_dir}")
    df = dask_cudf.read_parquet(data_dir)
    
    print(f"creating {k_mer}-mers...")
    if k_mer == 1:
        df = df.map_partitions(add_whitespace)
        df = df.map_partitions(split_whitespace)
    # elif (k_mer > 1) & (k_mer <= max_k_mer):
    #     df = df.map_partitions(get_kmers)
    #     df = df.map_partitions(split_whitespace)
    elif (k_mer > 1):
        df = df.map_partitions(get_kmers)
        df = df.map_partitions(split_whitespace)
    
    print(f"saving updated data to:\n{data_dir}")
    _ = df.to_parquet(data_dir)
    
    print("deleting dask df...")
    del df

    print("shutting down...")
    client.shutdown()
    client.close()
    print("label extraction complete!")