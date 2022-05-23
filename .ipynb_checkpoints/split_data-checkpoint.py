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
    random_seed = configs['random_seed']
    data_splits = configs['data_splits']
    
    data_splits_values = []
    for a_split, a_val in data_splits.items():
        data_splits_values.append(a_val)
    
    in_filepath = f"{output_dir}/{project_name}/data/{project_name}"
    print("reading data...")
    df = dask_cudf.read_parquet(in_filepath)
    
    print("splitting data...")
    df_list = df.random_split(data_splits_values, random_state=random_seed)

                           
    print("saving data splits...")
    for i, (a_split, a_val) in enumerate(data_splits.items()):
        out_filepath = f"{in_filepath}_{a_split}"
        _ = df_list[i].to_parquet(out_filepath)
    
    
    print("deleting dask df...")
    del df, df_list

    print("shutting down...")
    client.shutdown()
    client.close()
    print("label extraction complete!")