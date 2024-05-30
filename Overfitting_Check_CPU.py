import os
import GPy
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, median_absolute_error, mean_absolute_error
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

logging.basicConfig()
logger = logging.getLogger('GP-age')
logger.setLevel(logging.INFO)

def impute_KNN(array_with_missing):
    imputer = KNNImputer()
    imputed_array = imputer.fit_transform(array_with_missing)
    return imputed_array

def calc_stats(prediction, y):
    rmse = mean_squared_error(y, prediction, squared=False)
    med_ae = median_absolute_error(y, prediction)
    mean_ae = mean_absolute_error(y, prediction)
    return {'RMSE': rmse, 'MedAE': med_ae, 'MeanAE': mean_ae}

def output_results(average_results, output_dir):
    results_df = pd.DataFrame(average_results)
    if output_dir is not None:
        output_path = os.path.join(output_dir, 'GP-age_Average_Results.csv')
        results_df.to_csv(output_path, header=True, index=False, float_format='%.3f')
        logger.info(f'Results were saved to {output_path}')
    else:
        print('\n', results_df.round(3), '\n')

    return results_df

def visualize_results(results_df, output_dir):
    results_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Average Metrics Comparison')
    plt.ylabel('Values')
    plt.xticks(rotation=0)
    plt.legend(loc='best')
    if output_dir is not None:
        plot_path = os.path.join(output_dir, 'GP-age_Average_Results.png')
        plt.savefig(plot_path)
        logger.info(f'Visualization saved to {plot_path}')
    else:
        plt.show()

def load_and_preprocess_data(folder_path):
    all_ages = []
    all_methylation_data = []

    for file_path in Path(folder_path).glob('*.csv'):
        data = pd.read_csv(file_path, index_col=0, header=None)
        ages = data.iloc[0, :].astype(float).values  # Assuming the first row is the age
        methylation_data = data.iloc[1:, :].astype(float).T  # Assuming the rest is methylation data

        all_ages.append(ages)
        all_methylation_data.append(methylation_data)

    # Ensure all methylation datasets have the same features (intersection of all feature sets)
    common_features = set(all_methylation_data[0].columns)
    for methylation_data in all_methylation_data[1:]:
        common_features &= set(methylation_data.columns)

    if not common_features:
        raise ValueError("No common features found across datasets.")

    common_features = list(common_features)
    all_methylation_data = [df[common_features] for df in all_methylation_data]

    all_ages = np.concatenate(all_ages)
    all_methylation_data = np.concatenate([df.values for df in all_methylation_data])

    return all_ages, all_methylation_data

def age_match_data(ages1, data1, ages2, data2, span=5):
    def round_age(age):
        return span * round(age / span)

    ages1_rounded = np.array([round_age(age) for age in ages1])
    ages2_rounded = np.array([round_age(age) for age in ages2])

    matched_indices1 = []
    matched_indices2 = []

    unique_ages = np.unique(np.concatenate((ages1_rounded, ages2_rounded)))
    for age in unique_ages:
        indices1 = np.where(ages1_rounded == age)[0]
        indices2 = np.where(ages2_rounded == age)[0]
        min_count = min(len(indices1), len(indices2))
        matched_indices1.extend(np.random.choice(indices1, min_count, replace=False))
        matched_indices2.extend(np.random.choice(indices2, min_count, replace=False))

    matched_data1 = data1[matched_indices1]
    matched_data2 = data2[matched_indices2]
    matched_ages1 = ages1[matched_indices1]
    matched_ages2 = ages2[matched_indices2]

    return matched_data1, matched_ages1, matched_data2, matched_ages2

def train_and_test(data_folder_europe, data_folder_outside_europe, output_dir=None, iterations=35, age_span=5):
    logger.info('Starting training and testing procedure')

    average_results = {
        'Region': ['Europe_Test', 'Outside_Europe_Test'],
        'Average RMSE': [],
        'Average MedAE': [],
        'Average MeanAE': []
    }

    rmse_europe = []
    medae_europe = []
    mean_ae_europe = []
    rmse_outside_europe = []
    medae_outside_europe = []
    mean_ae_outside_europe = []

    for x in range(iterations):
        print(f"Iteration {x + 1}/{iterations}")
        # Load data
        ages_europe, methylation_data_europe = load_and_preprocess_data(data_folder_europe)
        ages_outside_europe, methylation_data_outside_europe = load_and_preprocess_data(data_folder_outside_europe)

        # Age match data
        X_europe, y_europe, X_outside_europe, y_outside_europe = age_match_data(
            ages_europe, methylation_data_europe, ages_outside_europe, methylation_data_outside_europe, span=age_span
        )

        X_europe = impute_KNN(X_europe)
        y_europe = y_europe.reshape(-1, 1)  # Reshaping to ensure it's a 2D array
        X_outside_europe = impute_KNN(X_outside_europe)
        y_outside_europe = y_outside_europe.reshape(-1, 1)  # Reshaping to ensure it's a 2D array

        # Split European data into training and testing sets
        X_train_europe, X_test_europe, y_train_europe, y_test_europe = train_test_split(
            X_europe, y_europe, test_size=0.2, random_state=42
        )

        # Create and optimize GPR model
        model = GPy.models.GPRegression(X_train_europe, y_train_europe)
        model.optimize()

        # Evaluate performance on European test set
        mean_predictions_europe, _ = model.predict(X_test_europe)
        stats_europe = calc_stats(mean_predictions_europe, y_test_europe)
        rmse_europe.append(stats_europe['RMSE'])
        medae_europe.append(stats_europe['MedAE'])
        mean_ae_europe.append(stats_europe['MeanAE'])

        # Evaluate performance on outside European test set
        mean_predictions_outside_europe, _ = model.predict(X_outside_europe)
        stats_outside_europe = calc_stats(mean_predictions_outside_europe, y_outside_europe)
        rmse_outside_europe.append(stats_outside_europe['RMSE'])
        medae_outside_europe.append(stats_outside_europe['MedAE'])
        mean_ae_outside_europe.append(stats_outside_europe['MeanAE'])

    # Calculate averages
    average_results['Average RMSE'] = [
        np.mean(rmse_europe),
        np.mean(rmse_outside_europe)
    ]
    average_results['Average MedAE'] = [
        np.mean(medae_europe),
        np.mean(medae_outside_europe)
    ]
    average_results['Average MeanAE'] = [
        np.mean(mean_ae_europe),
        np.mean(mean_ae_outside_europe)
    ]

    # Output results
    results_df = output_results(average_results, output_dir)
    
    # Visualize results
    visualize_results(results_df, output_dir)

if __name__ == '__main__':
    # Hardcoded paths
    data_folder_europe = ''  # Path to Europe datasets
    data_folder_outside_europe = ''  # Path to OutsideEurope datasets
    output_dir = ''  # Directory for output results

    # User inputs for iterations and age span
    iterations = 50
    age_span = 5

    train_and_test(data_folder_europe, data_folder_outside_europe, output_dir, iterations, age_span)
