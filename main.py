
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Avtive investment vs Pasive investment                                                        -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: if723286                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/if723286/Portafolio-Activo-Pasivo                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import os
import pandas as pd

# directory containing the csv files
directory = 'files'



columns = ['Ticker']
df_global = pd.DataFrame(columns=columns)

# loop through all the files in the directory
for filename in os.listdir(directory):
    # check if the file is a csv file
    if filename.endswith(".csv"):
        # read the csv file and store it in a data frame
        df = pd.read_csv(os.path.join(directory, filename), skiprows=2, usecols=["Ticker"])
        # append the data frame to the list
        df_global = pd.concat([df_global, df])
        
df_global


occurrences = df_global['Ticker'].value_counts()
occurrences

#### Eligire solamente las acciones que hayan aparecido las 25 veces en el NAFTRAC, en total son 34 acciones:
print(occurrences.head(33))