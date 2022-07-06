# Palm_Tree_Disease_Prediction

this folder contains:
* 5 running  python scripts (5 CNN models)
       to predict disease on palm tree leaves each has different tools and techiques and give different accuracy:
       they do  classification of three types of leaves 
       representing 3 categories :["Brown_spot","White_Scale","Healthy"] which themselves representing 
       the type of disease or being healthy.
* images : zipped folder containing the dataset for training and testing(tain  and test sub_folders), which they contain
           plam tree leaves classified into three categories under three sub_directories(brown_spots,healthy and white_scale)
           note:-> this zipped folder gets unzipped and saved in a folder with the same name in every run of any of the 
                  provided models(delete the created images folder with other generated .h5 and .model files after every execution )
          
* environment file: contains all packages (pip and conda packages) needs to be installed before runing the scripts.


NOTE :  1.launch the anaconda command prompt and navigate to the working directory 
        2.create an environment <new_env> using the environment.yml file 
       (place environment.yml file under the  same working directory with the rest of the files(images.7z and the python scripts)) 
        and then type the following command between brackets  in ur anaconda command line under the same working directory:
        [ conda env create --name new_env -f environment.yml  ]
        3. type the follwing command to activate the environment previously created
        [ conda activate new_env                              ]

Requirement: conda :   conda version : 4.8.3
                       conda-build version : 3.18.11
                       platform : win-64
             
