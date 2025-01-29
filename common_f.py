import os

import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import shap
import shap
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from scipy import interp, stats
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, \
                            auc, confusion_matrix, accuracy_score, f1_score, \
                            recall_score, precision_score, average_precision_score, \
                            balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

current_path = os.getcwd()

def prepare_data(csvPath, data, isOverSample=False,
                 outcome="outcome", oneHotCols=[], test_ratio=0.3,
                 show_heatmap=True, show_details=True, random_seed=2024):

    if data.isnull().sum().any():
        # TODO: Handle null values for specific dataset
        raise ValueError("Null values exist, please handle it first!")

    if len(oneHotCols) > 0:
        # NOTE:OneHotEncoder multiple categorical variables
        diet_column = 'Carnivorous or vegetarian preferences'
        if diet_column in oneHotCols:
            oneHotCols.remove(diet_column)
            diet_series = data[[diet_column]]
            diet_encoder = OneHotEncoder()
            diet_encoded_data = diet_encoder.fit_transform(diet_series).toarray()
            diet_columns = np.array(['Vegetable-heavy diet', 'Balanced diet', 'Meat-heavy diet'])
            diet_encoded_df = pd.DataFrame(diet_encoded_data, columns=diet_columns)
            data = data.drop(columns = [diet_column])
            data = pd.concat([data, diet_encoded_df], axis=1)
        if len(oneHotCols) > 0:
            encoder = OneHotEncoder()
            multi_categ_cols = data[oneHotCols]
            encoded_data = encoder.fit_transform(multi_categ_cols).toarray()
            encoded_df = pd.DataFrame(encoded_data, 
                                    columns = encoder.get_feature_names_out(oneHotCols))
            data = data.drop(columns = oneHotCols)
            data = pd.concat([data, encoded_df], axis = 1)
        data.to_csv(os.path.join(csvPath, 'dataSet_oneHot.csv'),
                    encoding = 'utf-8', index = None)
    
    if show_heatmap:
        plot_heatmap(data)
    
    X_train, X_test, y_train, y_test = train_test_split(data.drop(outcome, axis = 1),
                                                        data[outcome], 
                                                        test_size = test_ratio,
                                                        random_state = random_seed)
    
    if isOverSample:
        X_train.to_csv(os.path.join(csvPath, 'X_train_before_smote.csv'),
                       encoding = 'utf-8', index = None)
        y_train.to_csv(os.path.join(csvPath, 'y_train_before_smote.csv'),
                       encoding = 'utf-8', index = None)
        oversample = SMOTE(random_state = random_seed)
        X_train, y_train = oversample.fit_resample(X_train, y_train.ravel())

    if show_details:
        print('dataSet total size:\n', data.shape,
              '\ndataSet target:\n', data[outcome].value_counts(), 
              '\ntrainSet value_counts:\n', pd.DataFrame(y_train).value_counts(), 
              '\ntestSet value_counts:\n', pd.DataFrame(y_test).value_counts())
    
    pd.DataFrame(X_train).to_csv(os.path.join(csvPath, 'X_train.csv'),
                                 encoding = 'utf-8', index = None)
    pd.DataFrame(y_train).to_csv(os.path.join(csvPath, 'y_train.csv'),
                                 encoding = 'utf-8', index = None)
    X_test.to_csv(os.path.join(csvPath, 'X_test.csv'),
                  encoding = 'utf-8', index = None)
    y_test.to_csv(os.path.join(csvPath, 'y_test.csv'),
                  encoding = 'utf-8', index = None)

    return X_train, X_test, y_train, y_test

def plot_heatmap(data):
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize = (36,12), dpi = 1200)
    sns.heatmap(data.corr(), annot = True, cmap = 'Blues')
    plt.savefig(current_path + '/Results/headmap.png', dpi = 1200, 
                bbox_inches = 'tight', transparent = False)
    plt.savefig(current_path + '/Results/pdf/headmap.pdf', format='pdf',
                bbox_inches='tight')
    plt.show()

def plot_roc_curves(y_true_list, y_probs_list, labels, colors, fileName):

    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize = (3.25, 3), dpi = 2000)

    for i, (y_true, y_probs) in enumerate(zip(y_true_list, y_probs_list)):
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        roc_auc = roc_auc_score(y_true, y_probs)
        aucci = roc_confidence_interval(y_true, y_probs, alpha = 0.95)
        plt.plot(fpr, tpr, label ='{}: {:.3f} ({:.3f}-{:.3f})'.format\
                 (labels[i], roc_auc, aucci[1], aucci[2]),
                 color = colors[i], linewidth = 1)

    legend = plt.legend(title = "AUROC (95% CI)", loc = 4, 
                        bbox_to_anchor = (1, 0.03), borderaxespad = 0,
                        fontsize = 8, labelspacing = 0.6)
    legend._legend_box.align = "left"
    legend.get_frame().set_alpha(1)
    plt.setp(legend.get_title(), fontsize = 8, horizontalalignment = 'left')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    plt.ylabel('True Positive Rate', fontsize = 10)
    plt.xlabel('False Positive Rate', fontsize = 10)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],
               fontsize = 8)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],
               fontsize = 8)
    plt.xlim(-0.02, 1)    
    plt.ylim(0, 1)   
    plt.grid(axis='y')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(current_path + f'/Results/{fileName}.png', dpi = 2000, 
                bbox_inches = 'tight', transparent = False)
    plt.savefig(current_path + f'/Results/pdf/{fileName}.pdf', 
                format='pdf', bbox_inches='tight')
    plt.show()

def plot_multi_roc_curves(probs_paths, true_path, labels, colors, roc_avg, fileName):
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize = (3.25, 3), dpi = 1200)
    num_models = len(probs_paths)
    y_true_all = pd.read_csv(true_path, encoding = 'utf8')
    for i in range(0, num_models):
        tprs = []
        aucs = []
        base_fpr = np.linspace(0, 1, 101)
        y_pred_all = pd.read_csv(probs_paths[i], encoding = 'utf8')
        for j in range (0, y_pred_all.shape[1]):
            y_pred = y_pred_all.iloc[:-1, j]
            y_true = y_true_all.iloc[:-1, j]
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_pred)
            # plt.plot(fpr, tpr, colors[i][0], alpha = 0.15)
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs.append(tpr)
            aucs.append(roc_auc)

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis = 0)
        std_tprs = tprs.std(axis = 0)
        tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
        tprs_lower = np.maximum(mean_tprs - std_tprs, 0)

        aucs = np.array(aucs)
        mean_auc = aucs.mean()
        std_auc = aucs.std()
        if not roc_avg.empty:
            each_avg = roc_avg.loc[i, 'ROC AUC_mean']
            each_std = roc_avg.loc[i, 'ROC AUC_std']
            legend = labels[i]+': '+ f"{each_avg:.3f}" + '$\pm$'+ f"{each_std:.3f}"
        else:
            legend = labels[i] +': ' + str(np.round(mean_auc, 3)) + '$\pm$' + \
                             str(np.round(std_auc, 3))
        plt.plot(base_fpr, mean_tprs, colors[i][0], linewidth = 1, label = legend)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, 
                         color = colors[i][1], alpha = 0.2)
    
    legend = plt.legend(loc = 4, fontsize = 7, 
                        bbox_to_anchor = (1, 0.03),
                        borderaxespad = 0,
                        labelspacing = 0.6)
    legend.get_frame().set_alpha(1)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth = 1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('True Positive Rate', fontsize = 10)
    plt.xlabel('False Positive Rate', fontsize = 10)

    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],
               fontsize = 8)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],
               fontsize = 8) 
    plt.grid(axis='y')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(left=0, bottom=0) 
    plt.tight_layout()
    plt.savefig(current_path + f'/Results/{fileName}.png', 
                dpi = 1200, bbox_inches = 'tight', transparent = False)
    plt.savefig(current_path + f'/Results/pdf/{fileName}.pdf', 
                format='pdf', bbox_inches='tight')
    plt.show()


def plot_pr_curves(y_true_list, y_probs_list, labels, colors, fileName):

    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize = (3.25, 3), dpi = 2000)

    for i, (y_true, y_probs) in enumerate(zip(y_true_list, y_probs_list)):
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        average_precision = average_precision_score(y_true, y_probs)
        apci = ap_confidence_interval(y_true, y_probs, alpha = 0.95)
        plt.plot(recalls, precisions, label ='{}: {:.3f} ({:.3f}-{:.3f})'.format\
                 (labels[i], average_precision, apci[1], apci[2]),
                 color = colors[i], linewidth = 1)

    legend = plt.legend(title = "AP (95% CI)", loc = 4,
                        bbox_to_anchor = (1, 0.03), borderaxespad = 0,
                        fontsize = 8, labelspacing = 0.6)
    
    legend._legend_box.align = "left"
    legend.get_frame().set_alpha(1)
    plt.setp(legend.get_title(), fontsize = 8)
    plt.xlabel('Recall', fontsize = 10)
    plt.ylabel('Precision', fontsize = 10)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],
               fontsize = 8)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],
               fontsize = 8)
    plt.xlim(0, 1)    
    plt.ylim(0, 1)   
    plt.grid(axis='y')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(current_path + f'/Results/{fileName}.png', dpi = 2000, 
                bbox_inches = 'tight', transparent = False)
    plt.savefig(current_path + f'/Results/pdf/{fileName}.pdf', 
                format='pdf', bbox_inches='tight')
    plt.show()
        
def plot_shap(explainer, shap_values, X, sample_index=None, feature_name=None, fileName='shap'):
    shap.initjs()
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize = (4.7, 3), dpi = 1200)
    if sample_index is not None:
        shap.force_plot(explainer.expected_value, shap_values[sample_index,:], 
                        X.iloc[sample_index,:])
    elif feature_name is not None:
        shap.dependence_plot(feature_name, shap_values, X)
    else:
        # NOTE: Set shap variable counts here
        shap.summary_plot(shap_values, X, max_display = 23,
                          plot_size = (4.7, 5.9), show = False)
    
    ax = plt.gca()
    for label in ax.get_yticklabels():
        label.set_fontsize(8)
    for label in ax.get_xticklabels():
        label.set_fontsize(8)
    ax.xaxis.label.set_size(10)
    plt.tight_layout()
    plt.savefig(current_path + f'/Results/{fileName}.png',
                dpi = 1200, bbox_inches = 'tight', transparent = False)
    plt.savefig(current_path + f'/Results/pdf/{fileName}.pdf',
                format='pdf', bbox_inches='tight')
    plt.show()

def plot_importances(importances, names, model_type):
    feature_importance = np.array(importances)
    feature_names = np.array(names)
    data = {'feature_names':feature_names, 'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    # NOTE: Set importances counts here
    # fi_df = fi_df.head(10)
    fi_df.sort_values(by = ['feature_importance'], ascending = False, inplace = True)

    num_colors = len(fi_df['feature_names'])
    color_palette = sns.color_palette("Blues_r", n_colors = num_colors + 5)
    colors = color_palette[:num_colors]
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize = (6.25, 1.8), dpi = 2000)
    # NOTE: Change the histogram style here
    # sns.barplot(x = fi_df['feature_importance'], y = fi_df['feature_names'])
    sns.barplot(x = fi_df['feature_importance'], y = fi_df['feature_names'], 
                palette = colors)
    plt.title('Feature importance', fontsize = 8)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(current_path + '/Results/' + model_type + ' feature_importances.png', 
                dpi = 2000, bbox_inches = 'tight', transparent = False)
    plt.savefig(current_path + '/Results/pdf/' + model_type + ' feature_importances.pdf', 
                format='pdf', bbox_inches='tight')
    plt.show()

def evaluate_model(y_val, y_val_pred, y_val_proba):
    f1 = f1_score(y_val, y_val_pred)
    accuracy = accuracy_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_proba)

    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)

    metrics = {
        'F1 Score': f1,
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'ROC AUC': roc_auc,
        'Specificity': specificity,
        'Sensitivity': sensitivity,
        'Balanced accuracy': balanced_accuracy
    }

    return metrics

def roc_confidence_interval(y_true, y_scores, alpha = 0.95):

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    n1 = sum(y_true)
    n2 = len(y_true) - n1
    q1 = roc_auc / (2 - roc_auc)
    q2 = 2 * roc_auc**2 / (1 + roc_auc)
    se_auc = np.sqrt((roc_auc * (1 - roc_auc) +
                    (n1 - 1) * (q1 - roc_auc**2) +
                    (n2 - 1) * (q2 - roc_auc**2)) / (n1 * n2))
    z = stats.norm.ppf(alpha + (1 - alpha) / 2)
    lower = max(0, roc_auc - z * se_auc)
    upper = min(1, roc_auc + z * se_auc)

    return roc_auc, lower, upper

def ap_confidence_interval(y_true, y_scores, alpha=0.95):

    ap = average_precision_score(y_true, y_scores)
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    n = len(y_true)
    variance = (ap * (1 - ap)) / n
    se_ap = np.sqrt(variance)
    
    z = stats.norm.ppf(alpha + (1 - alpha) / 2)
    lower = max(0, ap - z * se_ap)
    upper = min(1, ap + z * se_ap)
    
    return ap, lower, upper

def calculate_sensitivity_specificity(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    return sensitivity, specificity