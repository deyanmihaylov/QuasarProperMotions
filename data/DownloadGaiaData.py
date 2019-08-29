from astroquery.utils.tap.core import Tap

gaiaURL = "http://gea.esac.esa.int/tap-server/tap"

query2 = "SELECT * FROM gaiadr2.gaia_source AS gaia WHERE gaia.frame_rotator_object_type = 2"
query3 = "SELECT * FROM gaiadr2.gaia_source AS gaia WHERE gaia.frame_rotator_object_type = 3"
query2and3 = "SELECT * FROM gaiadr2.gaia_source AS gaia WHERE gaia.frame_rotator_object_type = 2 OR gaia.frame_rotator_object_type = 3"

gaia = Tap(url=gaiaURL)

job2 = gaia.launch_job_async(query2, name=None, output_file="data/type2.csv",
                             output_format="csv", verbose=False,
                             dump_to_file=True, background=False,
                             upload_resource=None, upload_table_name=None)

job3 = gaia.launch_job_async(query3, name=None, output_file="data/type3.csv",
                             output_format="csv", verbose=False,
                             dump_to_file=True, background=False,
                             upload_resource=None, upload_table_name=None)

job2and3 = gaia.launch_job_async(query2and3, name=None, output_file="data/type2and3.csv",
                             output_format="csv", verbose=False,
                             dump_to_file=True, background=False,
                             upload_resource=None, upload_table_name=None)
