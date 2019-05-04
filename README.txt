This is a small repository created for the tSNE-UMAP day organized in Paul Francois' group on May 8, 2019. 

## FORMATTING DATA
Use the format_tools.py functions to format your data into the desired pandas.DataFrame structure. We want a DataFrame where each column is a different observable and each row is a different sample point. Rows can be multi-indexed with pandas to give more information about the conditions corresponding to each sample point. For instance, if you have data about the positions and velocities of N particles in a box, at different temperatures and pressures, the final DataFrame might look like this:

Observables                   vx  vy  vz   x   y    z
Feeling Temperature Pressure                         
cold    10 C        0 atm      0  20  40  60  80  100
                    1 atm      1  21  41  61  81  101
                    2 atm      2  22  42  62  82  102
                    3 atm      3  23  43  63  83  103
                    4 atm      4  24  44  64  84  104
        20 C        0 atm      5  25  45  65  85  105
                    1 atm      6  26  46  66  86  106
                    2 atm      7  27  47  67  87  107
                    3 atm      8  28  48  68  88  108
                    4 atm      9  29  49  69  89  109
hot     30 C        0 atm     10  30  50  70  90  110
                    1 atm     11  31  51  71  91  111
                    2 atm     12  32  52  72  92  112
                    3 atm     13  33  53  73  93  113
                    4 atm     14  34  54  74  94  114
        40 C        0 atm     15  35  55  75  95  115
                    1 atm     16  36  56  76  96  116
                    2 atm     17  37  57  77  97  117
                    3 atm     18  38  58  78  98  118
                    4 atm     19  39  59  79  99  119

The rows are indexed with a MultiIndex, to specify each experimental parameter for a given row. The columns could be a MultiIndex as well, to regroup some observables (for instance, a level could be added to regroup x and vx, y and vy, z and vz). 

Look at the API dcumentation to see the different functions available for formatting to this kind of DataFrame (df_from_ndarray and df_from_blocks). Of course, you are free to reformat your data manually if they do not suit your needs. 

Another, more realistic example is provided with single-cell RNA sequencing data from Hrvatin S. et al. Single-cell analysis of experience-dependent transcriptomic states in the mouse visual cortex. Nature Neurosci., 2018. It was used in a paper that we have read in JC, La Manno G. et al. RNA velocity of single cells. Nature, 2018. 

## DIMENSIONAL REDUCTION
Some functions might be provided in analyze_tools.py. Coming soon...
