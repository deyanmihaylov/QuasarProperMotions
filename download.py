from astroquery.gaia import Gaia

def download_Gaia_data(
    query: str,
    file_name: str,
) -> bool:
    output_file = f"./data/{file_name}"

    try:
        job = Gaia.launch_job_async(
            query,
            name="job",
            output_file=output_file,
            output_format="csv",
            verbose=False,
            dump_to_file=True,
            upload_resource=None,
            upload_table_name=None
        )
        job_success = True
    except:
        job_success = False

    return job_success

if __name__ == "__main__":
    queries = [
        "SELECT * FROM gaiadr2.gaia_source AS gaia WHERE \
            gaia.frame_rotator_object_type = 2",
        "SELECT * FROM gaiadr2.gaia_source AS gaia WHERE \
            gaia.frame_rotator_object_type = 3",
        "SELECT * FROM gaiadr2.gaia_source AS gaia WHERE \
            gaia.frame_rotator_object_type = 2 OR \
            gaia.frame_rotator_object_type = 3",
        "SELECT * FROM gaiaedr3.gaia_source AS edr3 WHERE \
            edr3.source_id IN (SELECT source_id FROM gaiaedr3.agn_cross_id)",
    ]

    file_names = [
        "type2.csv",
        "type3.csv",
        "type2and3.csv",
        "edr3.csv",
    ]

    for query, file_name in zip(queries, file_names):
        if download_Gaia_data(
            query,
            file_name
        ) == True:
            print(f"{file_name} was downloaded successfully.")
        else:
            print(f"There was a problem downloading {file_name}.")
