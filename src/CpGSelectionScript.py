import pandas as pd
from sklearn.impute import KNNImputer
import csv
import os
import glob

def select_columns(input_file, output_file, cpg_sites_to_keep):
    cpg_sites_written = set()

    with open(input_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip the original header
        header = next(reader) # Make the second row the header

        # Impute missing values for the entire file
        df = pd.read_csv(input_file, index_col=0)
        imputer = KNNImputer(n_neighbors=5)
        imputed_data = imputer.fit_transform(df.T)
        df_imputed = pd.DataFrame(imputed_data.T, columns=df.columns, index=df.index)

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)

        for cpg_site in cpg_sites_to_keep:
            if cpg_site in df_imputed.index:  # Write existing data
                writer.writerow([cpg_site] + list(df_imputed.loc[cpg_site]))
                cpg_sites_written.add(cpg_site)
            else:  # Impute and write for new rows
                imputed_values = imputer.fit_transform([df_imputed.mean(axis=0)])
                values = imputed_values[0].round(3)  # Round to 3 decimal places
                writer.writerow([cpg_site] + list(values))
                cpg_sites_written.add(cpg_site)
                
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file, index_col=0)

    # Separate the header row
    header = df.iloc[0]
    data = df.iloc[1:]

    # Sort the rows starting from the second row based on the first column
    sorted_data = pd.concat([header.to_frame().transpose(), data.sort_values(by=df.columns[0])])

    # Write the sorted DataFrame back to a CSV file
    sorted_data.to_csv('sorted_file.csv', index=False)

def main():
    input_folder = '' # Path to folder with input data
    output_folder = '' # Path to folder where you want to output the data
    cpg_sites_to_keep_all = [
        'cg16867657', 'cg22454769', 'cg06639320', 'cg04875128', 'cg19283806',
        'cg24724428', 'cg07553761', 'cg24079702', 'cg08128734', 'cg12934382',
        'cg08468401', 'cg20816447', 'cg00573770', 'cg06335143', 'cg06155229',
        'cg03032497', 'cg06619077', 'cg17804348', 'cg00329615', 'cg23479922',
        'cg10501210', 'cg19991948', 'cg27312979', 'cg23186333', 'cg25413977',
        'cg22078805', 'cg17621438', 'cg21878650', 'cg04503319', 'cg09809672'
    ]

    for input_file in glob.glob(os.path.join(input_folder, '*.csv')):
        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_folder, file_name)

        select_columns(input_file, output_file, cpg_sites_to_keep_all)

if __name__ == "__main__":
    main()
