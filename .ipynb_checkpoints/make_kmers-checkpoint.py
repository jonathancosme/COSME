from global_funcs import load_program_config
from dask.distributed import Client
from dask_cuda import LocalCUDACluster


if __name__ == '__main__':
    print("starting dask cluster...")
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print("finished starting dask cluster.")

    import dask_cudf

    print("loading configs...")
    configs = load_program_config()
    
    print("setting variables...")
    output_dir = configs['output_dir']
    project_name = configs['project_name']
    input_col_name = configs['input_col_name']
    random_seed = configs['random_seed']
    k_mer = configs['k_mer']
    possible_gene_values = configs['possible_gene_values']
    possible_gene_values = sorted(possible_gene_values)

    max_k_mer = 12
    
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

    in_filepath = f"{output_dir}/{project_name}/data/{project_name}"
    print(f"reading data from:\n{in_filepath}")
    df = dask_cudf.read_parquet(in_filepath)
    
    print(f"creating {k_mer}-mers...")
    if k_mer == 1:
        df = df.map_partitions(add_whitespace)
        df = df.map_partitions(split_whitespace)
    elif (k_mer > 1) & (k_mer <= max_k_mer):
        df = df.map_partitions(get_kmers)
        df = df.map_partitions(split_whitespace)
        
    out_filepath = in_filepath
    print(f"saving updated data to:\n{out_filepath}")
    _ = df.to_parquet(out_filepath)
    
    print("deleting dask df...")
    del df

    print("shutting down...")
    client.shutdown()
    client.close()
    print("label extraction complete!")