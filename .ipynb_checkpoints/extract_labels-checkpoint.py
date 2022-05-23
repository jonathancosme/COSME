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
    clean_fasta_filepath = configs['clean_fasta_file']
    output_dir = configs['output_dir']
    project_name = configs['project_name']
    base_col_names =  configs['base_col_names']
    label_col_name = configs['label_col_name']
    label_regex = configs['label_regex']
    random_seed = configs['random_seed']
    
    def extract_labels(df):
        df[label_col_name] = df[label_col_name].str.extract(label_regex).loc[:, 0]
        return df
    
    print("reading data...")
    df = dask_cudf.read_parquet(clean_fasta_filepath).repartition(partition_size="10M")
    print("extracting all labels...")
    df = df.map_partitions(extract_labels)
    
    
    print("getting unique labels...")
    unq_labs = df.sort_values(label_col_name)[label_col_name].unique().to_frame()
    out_filepath = f"{output_dir}/{project_name}/data/unq_labels" 
    print(f"saving unique labels to:\n{out_filepath}")
    _ = unq_labs.to_parquet(out_filepath)
    out_filepath = f"{output_dir}/{project_name}/data/unq_labels.csv" 
    print(f"saving unique labels to:\n{out_filepath}")
    _ = unq_labs.to_csv(out_filepath, index=False, single_file=True)
    
    print("encoding labels...")
    df = df.categorize(columns=[label_col_name])
    df[label_col_name] = df[label_col_name].cat.codes
    
    out_filepath = f"{output_dir}/{project_name}/data/{project_name}"
    print(f"saving updated data to:\n{out_filepath}")
    _ = df.to_parquet(out_filepath)
    
    print("deleting dask df...")
    del df

    print("shutting down...")
    client.shutdown()
    client.close()
    print("label extraction complete!")