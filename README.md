# Segment Land based on Irrigation Type from Remote Sensing Imageries

- *To prepare data:*
  - Go to 'Data_Prep' folder
  - run the "Save_tfrecords_WRLU_NoCDL_input_maker_7_4_l8l5_avg.py" file for each year (2003-2022, ignore 2012,2016,2017). This will save the tfrecords for each year in google drive
  - Now, get the files from the google drive in local server
  - Next run "Make_Training_Eval_data_on_WRLU_NoCDL_no_array_grid_minmax_using_library7_4_per_sensor_all.py" file for each year. It will process the tfrecords and save it as processed tftrecords


- *Train Data*
  - Models folder contains some Unet Models to run on the prepare data
